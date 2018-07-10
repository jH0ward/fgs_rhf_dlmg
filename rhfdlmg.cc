#include <typeinfo>
#include "rhfdlmg.h"
#include <psi4/psi4-dec.h>
#include <psi4/liboptions/liboptions.h>
#include <fstream>
#include <iostream>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libciomr/libciomr.h>
#include <psi4/libscf_solver/hf.h>
#include <psi4/libscf_solver/rhf.h>
#include <psi4/libmints/oeprop.h>
#include <psi4/psifiles.h>
#include <psi4/libpsi4util/PsiOutStream.h>
#include <psi4/libmints/matrix.h>
#include <psi4/libmints/factory.h>
#include <psi4/libpsi4util/process.h>
#include <psi4/libmints/molecule.h>
#include <psi4/libmints/basisset.h>
#include <psi4/libmints/mintshelper.h>
#include <psi4/libmints/petitelist.h>
#include <psi4/libmints/psimath.h>
#include <psi4/libqt/qt.h>
#include <psi4/libmints/integral.h>
#include <omp.h>
#include <chrono>
#include <stdio.h>
#include <iomanip>
#include "myelst.h"
#include <psi4/libfock/cubature.h>
#include <psi4/libfock/points.h>
#include <mkl.h>
#include <algorithm>
#include <cstring>



namespace psi{ namespace scf {

//#define DL_MG_INIT dl_mg_wrap_mp_dl_mg_init_wrap_
//#define DL_MG_SOLVER dl_mg_wrap_mp_dl_mg_solver_wrap_
//#define DL_MG_FREE dl_mg_mp_dl_mg_free_
//
#define DL_MG_INIT __dl_mg_wrap_MOD_dl_mg_init_wrap
#define DL_MG_SOLVER __dl_mg_wrap_MOD_dl_mg_solver_wrap
#define DL_MG_FREE __dl_mg_MOD_dl_mg_free

extern "C"
{
void DL_MG_INIT(int narray[3], double darray[3], int bc[3], int gstart[3],
                int gend[3], int * report_unit,int * max_iters);


void DL_MG_SOLVER(int sizes[3], double * pot, double * rho, int * order,
    int * report_unit, double * eps_full, double * eps_half,
    double * tol_val);

void DL_MG_FREE();
}

RHFDLMG::RHFDLMG(SharedWavefunction ref_wfn, Options
                  &options,std::shared_ptr<PSIO> psio): RHF(ref_wfn,
                  dynamic_cast<RHF*>(ref_wfn.get())->functional(), options,psio)
{
  nx_=options.get_int("nx");
  ny_=options.get_int("ny");
  nz_=options.get_int("nz");
  dlmg_tol=options.get_int("dlmg_tol");
  block_size_=options.get_int("block_size");
  narray[0]=nx_;
  narray[1]=ny_;
  narray[2]=nz_;
  rho0_=options.get_double("rho0");
  beta2_=2*options.get_double("beta");
  std::cout << "beta2 = " << beta2_ << "\n";
  std::cout << "Nx = " << nx_ << "\n";
  std::cout << "Ny = " << ny_ << "\n";
  std::cout << "Nz = " << nz_ << "\n";
  g_step_=options.get_double("g_step");

  darray[0]=g_step_;
  darray[1]=g_step_;
  darray[2]=g_step_;

  sigma_=options.get_double("sigma");
  eps_inf_=options.get_double("eps");
  outfile->Printf("eps = %15.10f\n",eps_inf_);
  basis_=ref_wfn->basisset();	
  V_eff_ = SharedMatrix (new Matrix("V eff",nirrep_,nsopi_,nsopi_));
  V_sgao_=SharedMatrix(new Matrix("V sgao",nirrep_,nsopi_,nsopi_));

  guess_Ca(ref_wfn->Ca_subset("SO","OCC"));
  guess_Cb(ref_wfn->Ca_subset("SO","OCC"));
  outfile->Printf("Just set the guesses and i'm printing them next\n");
  guess_Ca_->print_out();
  guess_Cb_->print_out();
  outfile->Printf("Done printing the guesses,calling rhf::guess()\n");

  RHF::guess();
  outfile->Printf("Back from RHF\n");

  centerx_=g_step_*(nx_-1)/2.0;
  centery_=g_step_*(ny_-1)/2.0;
  centerz_=g_step_*(nz_-1)/2.0;

	pot_sol_.resize(nx_*ny_*nz_);
  for (int i=0;i<nx_*ny_*nz_;i++){
    pot_sol_[i]=0.;
  }
  gmat_.resize(nx_*ny_*nz_);
  nmat_.resize(nx_*ny_*nz_);

  eps_half_.resize(nx_*ny_*nz_*3);
  eps_full_.resize(nx_*ny_*nz_);

  // Calculated permittivity or set it to 1 everywhere
  if (eps_inf_>1.0){
    calc_eps_symm();
  }
  else{
    for (int i=0;i<nx_*ny_*nz_*3;i++){
      eps_half_[i]=1.0;
    }
    for (int i=0;i<nx_*ny_*nz_;i++){
      eps_full_[i]=1.0;
    }
  }
  for (int i=0;i<nx_*ny_*nz_;i++) nmat_[i]=0.;
  calc_ndens_grid();

  std::cout << "Box is centered at " << centerx_ << ", " << centery_ << ", "
	          << centerz_ << std::endl;
  outfile->Printf("Printing alpha Density at the end of the constructor\n");
  Da_->print_out();
}
RHFDLMG::~RHFDLMG()
{
  std::cout << "Inside dtor\n" << std::flush;
}

void
RHFDLMG::density_block()
{
  outfile->Printf("Printing the density at the beginning of density_block()\n");
  Da_->print_out();
  std::cout << "Entering density block\n" << std::flush;
  std::cout << cube_ << "\n";
  int tnpt=nx_*ny_*nz_;
  double epsilon = 1.0e-12;


  // Get the AO density matrix and calculate the desymmetrized AO nbf x nbf
  // density matrix
  // In v.cc (line 121) D_AO_ is a vector of SharedMatrix and it's nbf x nbf
  // (nbf = the number of functions (either cartesian or spherical)
  // ! remember nao always refers to cartesian functions, ao2so the same
  //SharedMatrix Dao = shared_from_this()->D_subset_helper(shared_from_this()
  //    ->Da(), shared_from_this()->Ca(), "AO");

  
  // also see D_subset_helper in wavefunction.h
  //  D_subset helper takes AO (unsymmetrized pure) or CartAO (desymm Cart.)

  //SharedMatrix Dao_c1_cart = SharedMatrix(new Matrix("Dao unsymmetrized",
  //      nbf,nbf));

  //int block_size=6;
  int nblocks=nx_/block_size_;

  std::cout << "Using block_size of " << block_size_ << " gives nblocks " << 
                nblocks << "\n";
  std::vector<int> block_ends;
  for (int i=0;i<nblocks-1;i++){
    block_ends.push_back(block_size_*(i+1));
  }
  block_ends.push_back(nx_);
  std::cout << "Printing contents of block_ends...\n" << std::flush;
  for (int i=0;i<block_ends.size();i++){
    std::cout << block_ends[i] << "\t" << std::flush;
  }
  std::cout << "\n";


  int ilast=0;
  double tot_time =0.;
  int tot_pts=0;
  int remain=nx_%block_size_+block_size_;
  int i=0;
  std::vector<std::shared_ptr<PointFunctions>> workers;
  std::vector<SharedMatrix> DaoCart_vec;
  //
  // Begin block making code
  int npoints = nx_*ny_*nz_;
  int max_pts = block_size_*block_size_*block_size_;
  int max_functions=basisset_->nbf();
  double * gridx = new double [npoints];
  double * gridy = new double [npoints];
  double * gridz = new double [npoints];
  double * weight = new double [npoints];
  std::cout << "About to assign grid values\n" << std::flush;

  for (int indx=0;indx<npoints;indx++){
    int i=indx%nx_;
    int j=indx%(nx_*ny_)/nx_;
    int k=indx/nx_/ny_;
    gridx[indx]=g_step_*i-centerx_;
    gridy[indx]=g_step_*j-centery_;
    gridz[indx]=g_step_*k-centerz_;
    weight[indx]=1.0;
  }


  // Begin block making code
  // each block will be created from a new BlockOPoints consisting of
  // (n=number of points, gridx(y,z)[Q] = address of where the x (y,z) point is held
  //  at any point in the loop, &weight[Q]=all 1 for now, and extents=basisextents)
  std::shared_ptr<BasisExtents> extents(new BasisExtents(basis_,epsilon));;
  std::vector<std::shared_ptr<BlockOPoints>> blocks;
  std::cout << "Starting q loop\n" << std::flush;
  std::vector<int> block_indices;
  std::vector<int> block_ns;
  for (int Q=0;Q<npoints;Q+=max_pts){
    int n = (Q+max_pts >= npoints ? npoints - Q : max_pts);
    blocks.push_back(std::shared_ptr<BlockOPoints>(new BlockOPoints(n,&gridx[Q],
          &gridy[Q],&gridz[Q],&weight[Q],extents)));
    block_indices.push_back(Q);
    block_ns.push_back(n);
  }
//  std::cout << "Printing out contents of block_indices vector\n" << std::flush;
//  for (int i=0;i<block_indices.size();i++){
//    std::cout << block_indices[i] << "\t" << "\n";
//    std::cout << block_ns[i] << "\n";
//  }

  // End block making code
  //
  std::cout << "Ending q loop\n" << std::flush;

  SharedMatrix DaoCart= SharedMatrix(new Matrix(shared_from_this()->D_subset_helper(Da_,Ca_,"AO")));

  omp_set_dynamic(0);
  //int thread_var=24;
  //thread vs unthreaded number 
  int thread_var=omp_get_max_threads();
  //int thread_var=8;

  for (int i=0;i<nx_*ny_*nz_;i++){
    gmat_[i]=0.0;
  }

  for (int i=0;i<thread_var;i++){
        std::shared_ptr<PointFunctions> pworker(new RKSFunctions(basis_,max_pts,
          max_functions));
        pworker->set_ansatz(0);
        pworker->set_pointers(DaoCart);
        workers.push_back(pworker);
  }
  std::cout << "Through the thread container loop with length of workers = " << 
    workers.size() << "\n" << std::flush;
  mkl_set_num_threads(1);
  std::chrono::time_point<std::chrono::system_clock> blocks_start, blocks_end;
  blocks_start=std::chrono::system_clock::now();
  std::vector<long double> worker_density(thread_var);
  long double tot=0.;
#pragma omp parallel for reduction(+:tot) num_threads(thread_var)
  for (size_t Q=0;Q<blocks.size();Q++){
    int rank = omp_get_thread_num();
    std::shared_ptr<PointFunctions> pworker=workers[rank];
    std::shared_ptr<BlockOPoints> block = blocks[Q];
    pworker->compute_points(block);
    double * rho_a = pworker->point_value("RHO_A")->pointer();
    double tot_q=std::accumulate(rho_a,rho_a+block->npoints(),0.);

    tot+=tot_q;
    int starti=block_indices[Q];
    std::memcpy(&gmat_[starti],rho_a,block->npoints()*sizeof(double));

    if(block_ns[Q]!=block->npoints()){
      std::cout << "FAILURE BAD\n" << std::flush;
      exit(0);
    }
    
  }
  blocks_end=std::chrono::system_clock::now();
  std::chrono::duration<double> blocks_seconds=blocks_end-blocks_start;
  std::cout << "tot before scaling: " << std::setprecision(16) << tot << "\n" << std::flush;
  tot*=(g_step_*g_step_*g_step_);
  std::cout << "Computed total number of electrons = " << std::setprecision(14) << tot << "\n" << std::flush;
  std::cout << "Total calc. time for all blocks = " << blocks_seconds.count() << "\n";
  delete [] gridx;
  delete [] gridy;
  delete [] gridz;
  delete [] weight;

}


void
RHFDLMG::read_dens()
{
  std::ifstream densin(dens_file_);
  std::string myline;
  double read_val;
  for (int i=0;i<nirrep_;i++){
    std::getline(densin,myline);
    int last_col=0;
    while(last_col<nsopi_[i]){
      for (int j=0;j<3;j++){
        std::getline(densin,myline);
      }
      // Read and throw away row_id
      // Loop over nrows in irrep
      for (int r=0;r<nsopi_[i];r++){
        densin >> read_val;
        // Read in 5 values
        for (int j=last_col;j<std::min(last_col+5,nsopi_[i]);j++){
          densin >> read_val;
          Da_->set(i,r,j,read_val);
        }
        std::getline(densin,myline);
      }
      last_col+=5;
    }
    for (int j=0;j<3;j++){
      std::getline(densin,myline);
    }
  }
}


void
RHFDLMG::read_ca()
{
  std::ifstream casin(ca_file_);
  std::string myline;
  double read_val;
  for (int i=0;i<nirrep_;i++){
    std::getline(casin,myline);
    int last_col=0;
    // Read and throw away row_id
    while(last_col<nsopi_[i]){
      for (int j=0;j<3;j++){
        std::getline(casin,myline);
      }
      // Loop over nrows in irrep
      for (int r=0;r<nsopi_[i];r++){
        casin >> read_val;
        // Read in 5 values
        for (int j=last_col;j<std::min(last_col+5,nsopi_[i]);j++){
          casin >> read_val;
          Ca_->set(i,r,j,read_val);
        }
        std::getline(casin,myline);
      }
      last_col+=5;
    }
    for (int j=0;j<3;j++){
      std::getline(casin,myline);
    }
  }
}

double
RHFDLMG::compute_energy()
{
  std::cout << "about to call RHF::compute energy\n";
  double energy =  RHF::compute_energy();
  std::cout << " done with compute energy\n" << std::flush;
  std::cout << " swapping vectors\n" << std::flush;
  //exit(0);
  std::vector<double>().swap(pot_sol_);
  std::vector<double>().swap(gmat_);
  std::vector<double>().swap(nmat_);
  std::vector<double>().swap(eps_half_);
  std::vector<double>().swap(eps_full_);
  return energy;
}

double
RHFDLMG::compute_E()
{
  outfile->Printf("I'm in compute_E; gonna call RHF compute_E to start\n");
	double Etot=RHF::compute_E();
  outfile->Printf("Etot from RHF: %24.16f\n",Etot);
  
  double sg_pot = calc_sg_pot();
  Etot+=sg_pot;
  outfile->Printf("Interaction of cores with Veff = %24.16f\n",sg_pot); 

  double sgrep=calc_sg_rep();
  Etot+=sgrep;
  outfile->Printf("Returning total E = %24.16f\n",Etot);

	return Etot;
}

void RHFDLMG::form_H()
{
  RHF::form_H();
  outfile->Printf("Just called RHF::form_H()\nNow printing H_");
  H_->print_out();
  outfile->Printf("Calling V_sgao\n");
  calc_V_sgao_P();
  H_->add(V_sgao_);
  printf("Leaving my own form_H\n");
}

void RHFDLMG::form_G()
{
    outfile->Printf("Calling calc_V_sol from form_G\n");
    std::cout << "Iteration = " << iteration_ << "\n";
    // If iteration == 0, look for coeffs and density file
    if (iteration_==0){
      if (eps_inf_>1.0){
        ca_file_="sol_cas";
      }
      else{
        ca_file_="vac_cas";
      }
      std::ifstream casin(ca_file_.c_str());
      if (casin){
        std::cout << "cas file exists\n";
        // call read_density function
        read_ca();
        std::cout << "Printing out cas after read\n";
        Ca_->print_out();
      }
      else{
        std::cout << "cas file does not exist\n";
      }
      // repeat for density
      if (eps_inf_>1.0){
        dens_file_="sol_dens";
      }
      else{
        dens_file_="vac_dens";
      }
      std::ifstream densin(dens_file_.c_str());
      if (densin){
        std::cout << "dens file exists\n";
        // call read_density function
        read_dens();
        outfile->Printf("Printing out alpha density after file read\n");
        Da_->print_out();
      }
      else{
        std::cout << "density file does not exist\n";
      }
    }
    RHF::form_G();
    G_->axpy(-2.0,J_);
    calc_V_sol();
    outfile->Printf("Calling calc_V_eff from form_G\n");
    calc_V_eff();
    G_->axpy(2.0,V_eff_);
    J_->copy(V_eff_);
    outfile->Printf("Leaving my form_G\n");
    outfile->Printf("But I'm going to print out the coefficients first\n");
    Ca_->print_out();
}

void
RHFDLMG::calc_dens_grid()
{
  // This function made obsolete by density_block()
  //timer_init();
  std::shared_ptr<Molecule> molecule = Process::environment.molecule();
  Matrix geom=molecule->geometry();
  int natom=molecule->natom();
  molecule->print_in_bohr();
  Da_->print_out();
  int npts=nx_*ny_*nz_;
  std::cout << "about to set gmat and nmat\n";
  for (int i=0;i<npts;i++){
    gmat_[i]=0.;
    nmat_[i]=0.;
  }
  std::cout << "getting nao\n";
  int nao = basisset_->nao();
  std::cout << "counting nso\n";
  int nso=0;
  for (int h=0;h<nirrep_;h++) nso+=nsopi_[h];

	double tot=0.;
  std::cout << "nso = " << nso << "\n";
  double ndens=0;
  double rrmin=100;
  int savei,savej,savek;
  std::cout << "Beginning loop over x\n";
  double bfprod,xt,yt,zt;
  double Z[natom];
  double Zreal[natom];

  for (int a=0;a<natom;a++){
    Z[a]=molecule->Z(a);
    Zreal[a]=molecule->Z(a);
  }


  std::chrono::time_point<std::chrono::system_clock> dens_start, dens_end;
  dens_start = std::chrono::system_clock::now();
  double phiao[nao];
  double phiso[nso];
  int *col_offset = new int[nirrep_];
  MintsHelper helper(basis_, RHF::options_, 0);
  SharedMatrix aotoso = helper.petite_list(true)->aotoso();
  col_offset[0] = 0;
  for(int h=1; h < nirrep_; h++){
    col_offset[h] = col_offset[h-1] + aotoso->coldim(h-1);
  }
  double **u = block_matrix(nao, nso);
  for(int h=0; h < nirrep_; h++){
    for(int j=0; j < aotoso->coldim(h); j++){
      for(int ii=0; ii < nao; ii++){
        u[ii][j+col_offset[h]] = aotoso->get(h, ii, j);
      }
    }
  }
  #pragma omp parallel for reduction(+:tot,ndens) private(phiao,phiso)
  for (int i=0;i<nx_;i++){
    double xt=g_step_*i-centerx_;
    for (int j=0;j<ny_;j++){
      double yt=g_step_*j-centery_;
    for (int k=0;k<nz_;k++){
        double zt=g_step_*k-centerz_;
        int g=i+nx_*j+nx_*ny_*k;
        timer_on("computing b.f.");
        basisset_->compute_phi(phiao,xt,yt,zt);
        timer_off("computing b.f.");
        timer_on("Translate to symmetry basis");
        C_DGEMV('t',nao, nso, 1.0, &(u[0][0]), nso, &(phiao[0]), 
                1, 0.0, &(phiso[0]), 1);
        timer_off("Translate to symmetry basis");
        timer_on("Density on grid");
        for (int h=0;h<nirrep_;h++){
          for(int u=0; u < nsopi_[h]; u++){
            for(int v=0; v < u; v++){
            bfprod=(Da_->get(h,u,v))*4*phiso[u+col_offset[h]]*
                    phiso[v+col_offset[h]];
            gmat_[g]+=bfprod;
            }
          gmat_[g]+=(Da_->get(h,u,u))*2*phiso[u+col_offset[h]]*
                    phiso[u+col_offset[h]];
          }
        }
        timer_off("Density on grid");
        tot+=fabs(gmat_[g]);
        double rr;
        timer_on("Nuclear part");
        for (int l=0;l<natom;l++){
          double dx=xt-geom.get(l,0);
          double dy=yt-geom.get(l,1);
          double dz=zt-geom.get(l,2);
          rr=dx*dx+dy*dy+dz*dz;
          nmat_[g]-=1.0*Zreal[l]/pow(sigma_,3)*pow(M_PI,-1.5)*
                        exp(-1*rr/pow(sigma_,2));
          gmat_[g]-=1.0*Z[l]/pow(sigma_,3)*pow(M_PI,-1.5)*
                        exp(-1*rr/pow(sigma_,2));
          ndens+=Z[l]/pow(sigma_,3)*pow(M_PI,-1.5)*
                        exp(-1*rr/pow(sigma_,2));
        }
        timer_off("Nuclear part");
      }
    }
  }
  delete [] col_offset;
	std::cout << "total charge = " << std::setprecision(14) << tot*pow(g_step_,3) << std::endl;
	std::cout << "total nuc charge  = " << ndens*pow(g_step_,3) << std::endl;
  dens_end=std::chrono::system_clock::now();
  std::chrono::duration<double> dens_seconds=dens_end-dens_start;
  std::cout << "Calc_dens_grid took " << dens_seconds.count() << " seconds\n";
  timer_done();
}


void
RHFDLMG::calc_ndens_grid()
{
  // smear nuclear point charges over grid with tight Gaussians
  std::shared_ptr<Molecule> molecule = Process::environment.molecule();
  molecule->print_in_bohr();
  outfile->Printf("alpha Density to follow is from beginning of \
      calc_dens_grid\n");
  Da_->print_out();
  int npts=nx_*ny_*nz_;

  std::cout << "Beginning ndens grid loop over x\n";
  std::chrono::time_point<std::chrono::system_clock> dens_start, dens_end;
  dens_start = std::chrono::system_clock::now();
  int natom=molecule->natom();
  Matrix geom=molecule->geometry();
  double ndens=0.0;

  #pragma omp parallel for reduction(+:ndens) 
  for (int i=0;i<nx_;i++){
    double xt=g_step_*i-centerx_;
    double * Z = new double[natom];
    for (int a=0;a<natom;a++){
      Z[a]=molecule->Z(a);
    }
    for (int j=0;j<ny_;j++){
      double yt=g_step_*j-centery_;
      for (int k=0;k<nz_;k++){
        double zt=g_step_*k-centerz_;
        int g=i+nx_*j+nx_*ny_*k;
        double rr;
        for (int l=0;l<natom;l++){
          double dx=xt-geom.get(l,0);
          double dy=yt-geom.get(l,1);
          double dz=zt-geom.get(l,2);
          rr=dx*dx+dy*dy+dz*dz;
          nmat_[g]-=1.0*Z[l]/pow(sigma_,3)*pow(M_PI,-1.5)*
                        exp(-1*rr/pow(sigma_,2));
          ndens+=Z[l]/pow(sigma_,3)*pow(M_PI,-1.5)*
                        exp(-1*rr/pow(sigma_,2));
        }
      }
    }
    delete [] Z;
  }
	std::cout << "total nuc charge  = " << ndens*pow(g_step_,3) << std::endl;
  dens_end=std::chrono::system_clock::now();
  std::chrono::duration<double> dens_seconds=dens_end-dens_start;
  std::cout << "Calc_dens_grid took " << dens_seconds.count() << " seconds\n";
}

void RHFDLMG::calc_eps_symm()
{
  // Compute permittivity at full and mid-way grid points as functional of e- density
  // ( could be improved with density_block(), but the permittivity only needs to be
  //  computed once)
  timer_on("EPS");
  std::chrono::time_point<std::chrono::system_clock> eps_start, eps_end;
  eps_start = std::chrono::system_clock::now();
	double max_eps=0.;
	double tot=0;
	double min_eps=100.;

  // eps_half is permittivity at half-steps
	for (int i=0;i<nx_*ny_*nz_*3;i++){
	  eps_half_[i]=0.;
	}

  // eps_full is permittivity at full grid pts
  for (int i=0;i<nx_*ny_*nz_;i++){
    eps_full_[i]=0.;
  }

  int nao = basisset_->nao();
  int nso=0;

  // count up total symmetry orbitals
  for (int h=0;h<nirrep_;h++) nso+=nsopi_[h];

  // phiso will hold a.o. basis function values in s.o. basis
  double phiao[nao];
  double phiso[nso];

  // Get transformation matrix for A.O. -> S.O.
  int *col_offset = new int[nirrep_];

  MintsHelper helper(basis_, RHF::options_, 0);
  SharedMatrix aotoso = helper.petite_list(true)->aotoso();
  col_offset[0] = 0;
  for(int h=1; h < nirrep_; h++){
    col_offset[h] = col_offset[h-1] + aotoso->coldim(h-1);
  }

  double **u = block_matrix(nao, nso);
  for(int h=0; h < nirrep_; h++){
    for(int j=0; j < aotoso->coldim(h); j++){
      for(int ii=0; ii < nao; ii++){
        u[ii][j+col_offset[h]] = aotoso->get(h, ii, j);
      }
    }
  }

  #pragma omp parallel for private(phiao,phiso)
	for (int i=0;i<nx_;i++){
    double xt=g_step_*i-centerx_;
    for (int j=0;j<ny_;j++){
      double yt=g_step_*j-centery_;
      for (int k=0;k<nz_;k++){
        double zt=g_step_*k-centerz_;
        int findx=i+nx_*j+nx_*ny_*k;
        timer_on("EPS-comp_phi");
        // Compute all b.f. values at x,y,z
        basisset_->compute_phi(phiao,g_step_*i-centerx_,g_step_*j-centery_,
        g_step_*k-centerz_);
        timer_off("EPS-comp_phi");
        // Transform to so basis
        C_DGEMV('t',nao, nso, 1.0, &(u[0][0]), nso, &(phiao[0]), 1, 0.0,
                &(phiso[0]), 1);
        // Loop over each irrep, upper triangle of b.f. values, contracting with alpha-density
        double this_rho=0.;
        for (int h=0;h<nirrep_;h++){
          for(int u=0; u < nsopi_[h]; u++){
            for(int v=0; v < u; v++){
              double bfprod=(Da_->get(h,u,v))*4*phiso[u+col_offset[h]]*
                                              phiso[v+col_offset[h]];
              this_rho+=bfprod;
            }
            // Add diagonal
          this_rho+=(Da_->get(h,u,u))*2*phiso[u+col_offset[h]]*phiso[u+col_offset[h]];
          }
        }
        // Comptue permittivity given just computed density here
        eps_full_[findx] = 1.+(eps_inf_-1.)/2.0*(1.+(1.-pow((this_rho/rho0_),beta2_))/
             (1.+pow((this_rho/rho0_),beta2_)));

        // Repeat for all half-steps
        this_rho=0.;
        timer_on("EPS-comp_phi");
        basisset_->compute_phi(phiao,g_step_*(i+0.5)-centerx_,g_step_*j-centery_,
                              g_step_*k-centerz_);
        timer_off("EPS-comp_phi");
        C_DGEMV('t',nao, nso, 1.0, &(u[0][0]), nso, &(phiao[0]), 1, 0.0, &(phiso[0]), 1);
        for (int h=0;h<nirrep_;h++){
          for(int u=0; u < nsopi_[h]; u++){
            for(int v=0; v < u; v++){
              double bfprod=(Da_->get(h,u,v))*4*phiso[u+col_offset[h]]*
                                              phiso[v+col_offset[h]];
              this_rho+=bfprod;
            }
          this_rho+=(Da_->get(h,u,u))*2*phiso[u+col_offset[h]]*phiso[u+col_offset[h]];
          }
        }


        int indx=i+nx_*j+nx_*ny_*k+nx_*ny_*nz_*0;
        eps_half_[indx] = 1.+(eps_inf_-1.)/2.0*(1.+(1.-pow((this_rho/rho0_),beta2_))/
                       (1.+pow((this_rho/rho0_),beta2_)));


        this_rho=0.;
        timer_on("EPS-comp_phi");
        basisset_->compute_phi(phiao,g_step_*i-centerx_,g_step_*(j+0.5)-centery_,
                              g_step_*k-centerz_);
        timer_off("EPS-comp_phi");
        indx=i+nx_*j+nx_*ny_*k+nx_*ny_*nz_*1;
        C_DGEMV('t',nao, nso, 1.0, &(u[0][0]), nso, &(phiao[0]), 1, 0.0, &(phiso[0]), 1);
        for (int h=0;h<nirrep_;h++){
          for(int u=0; u < nsopi_[h]; u++){
            for(int v=0; v < u; v++){
              double bfprod=(Da_->get(h,u,v))*4*phiso[u+col_offset[h]]*
                                              phiso[v+col_offset[h]];
              this_rho+=bfprod;
            }
          this_rho+=(Da_->get(h,u,u))*2*phiso[u+col_offset[h]]*phiso[u+col_offset[h]];
          }
        }

        eps_half_[indx] = 1.+(eps_inf_-1.)/2.0*(1.+(1.-pow((this_rho/rho0_),beta2_))/
                       (1.+pow((this_rho/rho0_),beta2_)));

        this_rho=0.;
        timer_on("EPS-comp_phi");
        basisset_->compute_phi(phiao,g_step_*i-centerx_,g_step_*j-centery_,
                            g_step_*(k+0.5)-centerz_);
        timer_off("EPS-comp_phi");
        indx=i+nx_*j+nx_*ny_*k+nx_*ny_*nz_*2;
        C_DGEMV('t',nao, nso, 1.0, &(u[0][0]), nso, &(phiao[0]), 1, 0.0, &(phiso[0]), 1);
        for (int h=0;h<nirrep_;h++){
          for(int u=0; u < nsopi_[h]; u++){
            for(int v=0; v < u; v++){
              double bfprod=(Da_->get(h,u,v))*4*phiso[u+col_offset[h]]*
                                              phiso[v+col_offset[h]];
              this_rho+=bfprod;
            }
          this_rho+=(Da_->get(h,u,u))*2*phiso[u+col_offset[h]]*phiso[u+col_offset[h]];
          }
        }

        eps_half_[indx] = 1.+(eps_inf_-1.)/2.0*(1.+(1.-pow((this_rho/rho0_),beta2_))/
                       (1.+pow((this_rho/rho0_),beta2_)));
      }
    }
	}
	std::cout << "done looping over grid pts \n" << std::endl;
	for (int i=0;i<nx_*ny_*nz_;i++){
    if (eps_full_[i]>max_eps) max_eps=eps_full_[i];
    if (eps_full_[i]<min_eps) min_eps=eps_full_[i];
	}
	std::cout << "Max eps = " << max_eps << ", ";
	std::cout << "Min eps = " << min_eps << std::endl;
  eps_end=std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds=eps_end-eps_start;
  std::cout << "Calc_eps took " << elapsed_seconds.count() << " seconds\n";
  timer_off("EPS");
}

void RHFDLMG::calc_V_sol()
{
  // Calculate (total) electrostatic potential on grid using DL_MG library
  int bc[3] = { 2, 2, 2};
  int gstart[3] = {1,1,1};
  int gend[3] = {nx_,ny_,nz_};
  int maxcyc = 40;
  int rptu = 160; 
  const char* pathc;
  std::string path;
  if (pathc = std::getenv("TMPDIR")){
    path=pathc;
  }
  else {
    pathc = std::getenv("PWD");
    path=pathc;
  }
  std::string logname = "/dlmg_log.vac";
  if (eps_inf_ > 2.0){
    logname="/dlmg_log.sol";
  }
  std::string movename = "/dlmg_log.old";
  path.append(logname);
  const char * finalname = path.c_str();
  int plen = strlen(finalname);
  std::ifstream look4thisfile(std::getenv("TMPDIR")+std::string("/espao")
       ,std::ios::binary);
  if (!look4thisfile){
  compute_esp_ao();
  }
  compute_esp();
  density_block();
  // Calc_dens_grid obsolete, but leave if e- density computation needs to be debugged
  //calc_dens_grid();
  //
  // Now add gmat_ to nmat_
  // Temporary comment
  for (int i=0;i<nx_*ny_*nz_;i++){
   gmat_[i]+=nmat_[i];
  }
  std::cout << "bound_[0] = ";
  std::cout << pot_sol_[0] << "\n";

  double tol_val=pow(10.,-1*dlmg_tol);
  int max_iters=100;

  DL_MG_INIT(narray,darray,bc,gstart,gend,&rptu,
           &max_iters);


  int order=12; 

  std::cout << "Before dl_mg_solver" << std::endl;
  std::cout << "Calling solver now\n";

   DL_MG_SOLVER(narray,&pot_sol_[0], &gmat_[0],&order,&rptu,
       &eps_full_[0],&eps_half_[0],&tol_val);

  std::cout << "Done calling solver now\n" << std::flush;
  
  DL_MG_FREE();
}


double RHFDLMG::calc_sg_pot()
{
  // calculate interaction of smeared Gaussian nuclei with potential
  double tot=0.;
  for (int i=0;i<nx_;i++){
    for (int j=0;j<ny_;j++){
      for (int k=0;k<nz_;k++){
        int indx=i+nx_*j+nx_*ny_*k;
        tot+=nmat_[indx]*pot_sol_[indx];
      }
    }
  }
  tot/=2;
  tot*=(g_step_*g_step_*g_step_);
  return tot;
}

double RHFDLMG::calc_sg_rep()
{
  // Calculate the smeared representation nuclear repulsion
  double tot=0.;
  std::shared_ptr<Molecule> molecule=Process::environment.molecule();
  Matrix geom=molecule->geometry();
  int natom=molecule->natom();
  double * Z = new double [natom];
  for (int a=0;a<natom;a++){
    Z[a]=molecule->Z(a);
  }
  double f1=-1/sqrt(2*M_PI);
  double t1=0.;
	for (int l=0;l<natom;l++){
    t1+=Z[l]*Z[l];
	}
  t1=t1*f1*1/sigma_;
  double t2=0;
  for (int l=0;l<natom;l++){
    for (int m=0;m<l;m++){
      double dx=geom.get(l,0)-geom.get(m,0);
      double dy=geom.get(l,1)-geom.get(m,1);
      double dz=geom.get(l,2)-geom.get(m,2);
      double r2=dx*dx+dy*dy+dz*dz;
      double r=sqrt(r2);
      double signorm=sqrt(2*sigma_*sigma_);
      t2+=Z[l]*Z[m]/r*erf(r/signorm);
    }
  }
  delete [] Z;
  t2*=-1;
  return t1+t2;
}

void RHFDLMG::calc_V_eff()
{
  // Integrate potential into Fock matrix
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
	V_eff_->zero();
  int nthread=omp_get_max_threads();

  std::vector< SharedMatrix > thread_mats(nthread);
  for (int i=0;i<nthread;i++){
    thread_mats[i]=SharedMatrix(new Matrix(nirrep_,nsopi_,nsopi_));
    thread_mats[i]->zero();
  }

  MintsHelper helper(basis_,RHF::options_,0);
  SharedMatrix aotoso = helper.petite_list(true)->aotoso();
  int *col_offset = new int [nirrep_];
  col_offset[0]=0;
  for (int h=1;h<nirrep_;h++){
    col_offset[h]=col_offset[h-1]+aotoso->coldim(h-1);
  }

  int nao=basisset_->nao();
  int nso=nso_;
  double **u = block_matrix(nao,nso);
  for(int h=0; h < nirrep_; h++){
    for(int j=0; j < aotoso->coldim(h); j++){
      for(int i=0; i < nao; i++){
          u[i][j+col_offset[h]] = aotoso->get(h, i, j);
      }
    }
  }

  #pragma omp parallel for schedule(guided)
  for (int i=0;i<nx_;i++){
    int myid=omp_get_thread_num();
    double * phiao = new double [nao];
    double * phiso = new double [nso];
    double xt=g_step_*i-centerx_;
    for (int j=0;j<ny_;j++){
      double yt=g_step_*j-centery_;
      for (int k=0;k<nz_;k++){
        int indx=i+nx_*j+nx_*ny_*k;
        double zt=g_step_*k-centerz_;
        basisset_->compute_phi(phiao,xt,yt,zt);
        C_DGEMV('t',nao,nso,1.0,&(u[0][0]),nso,&(phiao[0]),1,0.0,
                &(phiso[0]),1);
        for (int h=0;h<nirrep_;h++){
          for (int u=0;u<nsopi_[h];u++){
            for (int v=0;v<=u;v++){
              double val=phiso[u+col_offset[h]]*phiso[v+col_offset[h]]
                        *pot_sol_[indx];
              thread_mats[myid]->add(h,u,v,val);
            }
          }
        }
      }
    }
  delete [] phiao;
  delete [] phiso;
  }
  // Fill in rest of symmetric matrix
  for (int h=0;h<nirrep_;h++){
    for (int u=0;u<nsopi_[h];u++){
      for (int v=0;v<u;v++){
        for (int t=0;t<nthread;t++){
          V_eff_->add(h,u,v,thread_mats[t]->get(h,u,v));
          V_eff_->add(h,v,u,thread_mats[t]->get(h,u,v));
        }
      }
    }
  }
  for (int h=0;h<nirrep_;h++){
    for (int u=0;u<nsopi_[h];u++){
      for (int t=0;t<nthread;t++){
        V_eff_->add(h,u,u,thread_mats[t]->get(h,u,u));
      }
    }
  }

	V_eff_->scale(pow(g_step_,3));
  V_eff_->scale(0.5);
	V_eff_->print_out();
  delete [] col_offset;
  std::cout << "leaving v eff calc" << std::endl;
  end=std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds=end-start;
  std::cout << "Calc_V_eff took " << elapsed_seconds.count() << " seconds\n";
}

void RHFDLMG::compute_esp()
{
  // Compute potential on boundary points analytically
  std::chrono::time_point<std::chrono::system_clock> esp_start, esp_end,z_end,y_end,x_end,write_end;
  esp_start = std::chrono::system_clock::now();

  int nbf = basisset_->nbf();

  FILE *gridout = fopen("grid_esp.dat", "w");
  if(!gridout)
      throw PSIEXCEPTION("Unable to write to grid_esp.dat");


  SharedMatrix Dtot = shared_from_this()->D_subset_helper(shared_from_this()->Da(), shared_from_this()->Ca(), "AO");
  Dtot->scale(2.0);
  std::ifstream ifile(std::getenv("TMPDIR")+std::string("/espao"),
      std::ios::binary);
  SharedMatrix ints(new Matrix("Ex integrals", nbf, nbf));
  ints->zero();
  Vector3 origin;
  double * read_value = new double[1];
  for (int i=0;i<nx_;i++){
    double xt=g_step_*i-centerx_;
    std::shared_ptr<Molecule> mol = basisset_->molecule();
    origin[0]=xt;
    int myid=omp_get_thread_num();
    for (int j=0;j<ny_;j++){
      double yt=g_step_*j-centery_;
      origin[1]=yt;
      for (int k=0;k<nz_;k=k+nz_-1){
        double zt=g_step_*k-centerz_;
        origin[2]=zt;
        int g=i+nx_*j+nx_*ny_*k;
        // Read in AO integrals
        for (int row=0;row<nbf;row++){
          for (int col=0;col<=row;col++){
            ifile.read((char *)read_value,sizeof(double));
            ints->set(row,col,*read_value);
            ints->set(col,row,*read_value);
          }
        }
        // Contract density matrix with AOxAO integrals matrix
        double Velec=Dtot->vector_dot(ints);
        double Vnuc = 0.0;
        int natom = mol->natom();
        for(int a=0; a < natom; a++) {
            Vector3 dR = origin - mol->xyz(a);
            double r = dR.norm();
            if(r > 1.0E-8) Vnuc += mol->Z(a)/r;
        }
        // coleman look here for bound change
        pot_sol_[g]=-1.*Velec-Vnuc;
      }
    }
  }
  z_end=std::chrono::system_clock::now();
  std::chrono::duration<double> z_seconds=z_end-esp_start;
  std::cout << "Z faces took " << z_seconds.count() << " seconds\n";
  ints->zero();
  std::shared_ptr<Molecule> mol = basisset_->molecule();
  for (int i=0;i<nx_;i++){
    double xt=g_step_*i-centerx_;
    origin[0]=xt;
    for (int k=0;k<nz_;k++){
      double zt=g_step_*k-centerz_;
      origin[2]=zt;
      for (int j=0;j<ny_;j=j+ny_-1){
        double yt=g_step_*j-centery_;
        origin[1]=yt;
        int g=i+nx_*j+nx_*ny_*k;
        // Read in AO integrals
        for (int row=0;row<nbf;row++){
          for (int col=0;col<=row;col++){
            ifile.read((char *)read_value,sizeof(double));
            ints->set(row,col,*read_value);
            ints->set(col,row,*read_value);
          }
        }
        double Velec=Dtot->vector_dot(ints);
        double Vnuc = 0.0;
        int natom = mol->natom();
        // TODO testing time required for nuclear V
        for(int a=0; a < natom; a++) {
            Vector3 dR = origin - mol->xyz(a);
            double r = dR.norm();
            if(r > 1.0E-8)
                Vnuc += mol->Z(a)/r;
        }
        // coleman look here for bound change
        pot_sol_[g]=-1.*Velec-Vnuc;
      }
    }
  }
  y_end=std::chrono::system_clock::now();
  std::chrono::duration<double> y_seconds=y_end-z_end;
  std::cout << "y faces took " << y_seconds.count() << " seconds\n";
  for (int j=0;j<ny_;j++){
    double yt=g_step_*j-centery_;
    origin[1]=yt;
    for (int k=0;k<nz_;k++){
      double zt=g_step_*k-centerz_;
      origin[2]=zt;
      for (int i=0;i<nx_;i=i+nx_-1){
        int g=i+nx_*j+nx_*ny_*k;
        double xt=g_step_*i-centerx_;
        origin[0]=xt;
        // read in ao integrals
        for (int row=0;row<nbf;row++){
          for (int col=0;col<=row;col++){
            ifile.read((char *)read_value,sizeof(double));
            ints->set(row,col,*read_value);
            ints->set(col,row,*read_value);
          }
        }
        double Velec=Dtot->vector_dot(ints);
        double Vnuc = 0.0;
        int natom = mol->natom();
        for(int a=0; a < natom; a++) {
            Vector3 dR = origin - mol->xyz(a);
            double r = dR.norm();
            Vnuc += mol->Z(a)/r;
        }
        // coleman look here for bound change
        pot_sol_[g]=-1.*Velec-Vnuc;
      }
    }
  }
  x_end=std::chrono::system_clock::now();
  std::chrono::duration<double> x_seconds=x_end-y_end;
  std::cout << "x faces took " << x_seconds.count() << " seconds\n";
  for (int g=0;g<nx_*ny_*nz_;g++){
    pot_sol_[g]/=eps_inf_;
  }
  for (int i=0;i<nx_;i++){
   for (int j=0;j<ny_;j++){
     for (int k=0;k<nz_;k=k+nz_-1){
       int indx = i+nx_*j+nx_*ny_*k;
       fprintf(gridout, "%16.10f\n", pot_sol_[indx]);
     }
   }
  }

  for (int i=0;i<nx_;i++){
   for (int j=0;j<ny_;j=j+ny_-1){
     for (int k=0;k<nz_;k++){
       int indx = i+nx_*j+nx_*ny_*k;
       fprintf(gridout, "%16.10f\n", pot_sol_[indx]);
     }
   }
  }

  for (int i=0;i<nx_;i=i+nx_-1){
   for (int j=0;j<ny_;j++){
     for (int k=0;k<nz_;k++){
       int indx = i+nx_*j+nx_*ny_*k;
       fprintf(gridout, "%16.10f\n", pot_sol_[indx]);
     }
   }
  }

  write_end=std::chrono::system_clock::now();
  std::chrono::duration<double> write_seconds=write_end-x_end;
  std::cout << "write faces took " << write_seconds.count() << " seconds\n";
  fclose(gridout);
  esp_end=std::chrono::system_clock::now();
  std::chrono::duration<double> esp_seconds=esp_end-esp_start;
  std::cout << "Calc_esp_vals took " << esp_seconds.count() << " seconds\n";
}

void RHFDLMG::compute_esp_ao()
{
  // Compute the electrostatic potential integrals in AO basis and save for 
  //   future iterations
  std::chrono::time_point<std::chrono::system_clock> esp_start, esp_end,z_end,y_end,x_end,write_end;
  esp_start = std::chrono::system_clock::now();


  outfile->Printf( "\n Electrostatic potential computed on the grid and written to grid_esp.dat\n");


  int nbf = basisset_->nbf();

  int nthread=omp_get_max_threads();
  std::vector< SharedMatrix> thread_mats(nthread);
  //std::cout << "Creating myelstin shared_ptr\n";
  //std::cout << "getting bs1\n";
  std::shared_ptr<BasisSet> mybs1 = integral_->basis1();

  std::vector< std::shared_ptr<MyElstInt>> epot_vec(nthread);
  std::cout << "creating vector of electrostatic integrals of size " << 
    nthread << "\n" << std::flush;
  for (int i=0;i<nthread;i++){
    epot_vec[i]=std::shared_ptr<MyElstInt>((dynamic_cast<MyElstInt*>(new MyElstInt(integral_->spherical_transform(),mybs1,mybs1,0))));
  }

  SharedMatrix Dtot = shared_from_this()->D_subset_helper(shared_from_this()->Da(), shared_from_this()->Ca(), "AO");
  Dtot->scale(2.0);
  std::ofstream ofile(std::getenv("TMPDIR")+std::string("/espao"),
        std::ios::binary);
  timer_on("Z faces");
  #pragma omp parallel for ordered schedule(static,1)
    for (int i=0;i<nx_;i++){
      double xt=g_step_*i-centerx_;
      double * buf4=new double [ny_*(nbf*(nbf+1))];
      SharedMatrix ints(new Matrix("Ex integrals", nbf, nbf));
      std::shared_ptr<Molecule> mol = basisset_->molecule();
      int natom=mol->natom();
      Vector3 origin;
      int myid=omp_get_thread_num();
      double Vnuc=0.;
      double Velec=0.;
      for (int j=0;j<ny_;j++){
        double yt=g_step_*j-centery_;
        for (int k=0;k<nz_;k=k+nz_-1){
          double zt=g_step_*k-centerz_;
          int g=i+nx_*j+nx_*ny_*k;
          origin[0]=xt;
          origin[1]=yt;
          origin[2]=zt;
          ints->zero();
          int k2=k;
          if (k2>0){
            k2=1;
          }
          if (myid==0) timer_on("Elst ints");
          epot_vec[myid]->compute(ints,origin);
          if (myid==0) timer_off("Elst ints");
          for (int row=0;row<nbf;row++){
            for (int col=0;col<=row;col++){
              int indx=j*2*(nbf*(nbf+1))/2+k2*(nbf*(nbf+1))/2+(row*(row+1))/2+col;
              buf4[indx]=ints->get(row,col);
            }
          }
        }
      }
      #pragma omp ordered
      for (int j=0;j<ny_;j++){
        for (int k=0;k<2;k++){
          for (int row=0;row<nbf;row++){
            for (int col=0;col<=row;col++){
              if (myid==0) timer_on("Indexing");
              int indx=j*2*(nbf*(nbf+1))/2+k*(nbf*(nbf+1))/2+(row*(row+1))/2+col;
              if (myid==0) timer_off("Indexing");
              if (myid==0) timer_on("Writing");
              ofile.write((char*) &buf4[indx],sizeof(double)); 
              if (myid==0) timer_off("Writing");
            }
          }
        }
      }
      delete [] buf4;
    }
    z_end=std::chrono::system_clock::now();
    std::chrono::duration<double> z_seconds=z_end-esp_start;
    std::cout << "Z faces took " << z_seconds.count() << " seconds\n" << std::flush;
    timer_off("Z faces");
    timer_done();
    std::cout << "Entering y faces \n";
    #pragma omp parallel for ordered schedule(static,1)
    for (int i=0;i<nx_;i++){
      double xt=g_step_*i-centerx_;
      double * buf4=new double [nz_*nbf*(nbf+1)];
      std::shared_ptr<Molecule> mol = basisset_->molecule();
      SharedMatrix ints(new Matrix("Ex integrals", nbf, nbf));
      int myid=omp_get_thread_num();
      Vector3 origin;
      for (int k=0;k<nz_;k++){
        double zt=g_step_*k-centerz_;
        for (int j=0;j<ny_;j=j+ny_-1){
          double yt=g_step_*j-centery_;
          origin[0]=xt;
          origin[1]=yt;
          origin[2]=zt;
          ints->zero();
          int j2=j;
          if (j2>0){
            j2=1;
          }
          epot_vec[myid]->compute(ints,origin);
          for (int row=0;row<nbf;row++){
            for (int col=0;col<=row;col++){
              int indx=k*2*(nbf*(nbf+1))/2+j2*(nbf*(nbf+1))/2+(row*(row+1))/
                  2+col;
              buf4[indx]=ints->get(row,col);
            }
          }
        }
      }
      #pragma omp ordered
        for (int k=0;k<nz_;k++){
          for (int j=0;j<2;j++){
            for (int row=0;row<nbf;row++){
              for (int col=0;col<=row;col++){
                int indx=k*2*(nbf*(nbf+1))/2+j*(nbf*(nbf+1))/2+
                      (row*(row+1))/2+col;
                ofile.write((char*) &buf4[indx],sizeof(double));
              }
            }
          }
        }
      delete [] buf4;
    }
    y_end=std::chrono::system_clock::now();
    std::chrono::duration<double> y_seconds=y_end-z_end;
    std::cout << "y faces took " << y_seconds.count() << " seconds\n";
    std::cout << "entering x faces\n";
    #pragma omp parallel for ordered schedule(static,1)
      for (int j=0;j<ny_;j++){
        double yt=g_step_*j-centery_;
        double * buf4=new double[nz_*(nbf*(nbf+1))];
        std::shared_ptr<Molecule> mol = basisset_->molecule();
        SharedMatrix ints(new Matrix("Ex integrals", nbf, nbf));
        int myid=omp_get_thread_num();
        Vector3 origin;
        for (int k=0;k<nz_;k++){
          double zt=g_step_*k-centerz_;
          for (int i=0;i<nx_;i=i+nx_-1){
            double xt=g_step_*i-centerx_;
            origin[0]=xt;
            origin[1]=yt;
            origin[2]=zt;
            ints->zero();
            int i2=i;
            if (i2>0){
              i2=1;
            }
            epot_vec[myid]->compute(ints,origin);
            for (int row=0;row<nbf;row++){
              for (int col=0;col<=row;col++){
                int indx=k*2*(nbf*(nbf+1))/2+i2*(nbf*(nbf+1))/2+(row*(row+1))/2
                  +col;
                buf4[indx]=ints->get(row,col);
              }
            }
          }
        }
        #pragma omp ordered
          for (int k=0;k<nz_;k++){
            for (int i=0;i<2;i++){
              for (int row=0;row<nbf;row++){
                for (int col=0;col<=row;col++){
                  int indx=k*2*(nbf*(nbf+1))/2+i*(nbf*(nbf+1))/2+(row*(row+1))/2
                          +col;
                  ofile.write((char*)&buf4[indx],sizeof(double));
                }
              }
            }
          }
        delete [] buf4;
      }
    x_end=std::chrono::system_clock::now();
    std::chrono::duration<double> x_seconds=x_end-y_end;
    std::cout << "x faces took " << x_seconds.count() << " seconds\n";
    esp_end=std::chrono::system_clock::now();
    std::chrono::duration<double> esp_seconds=esp_end-esp_start;
    std::cout << "Calc_esp_aos took " << esp_seconds.count() << " seconds\n";
}

void RHFDLMG::calc_V_sgao_P()
{
  // Calculate the smeared Gaussian analog of V_nuc and integrate into AO Basis
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  std::cout << "entering sgVaoP calc" << std::endl;
	V_sgao_->zero();
  std::shared_ptr<Molecule> molecule=Process::environment.molecule();
  Matrix geom=molecule->geometry();
  int natom=molecule->natom();
  double Z[natom];
  for (int a=0;a<natom;a++){
    Z[a]=molecule->Z(a);
  }

  const int nthread=omp_get_max_threads();
  //const int nthread=1;
  std::cout << "num threads = " << nthread << std::endl;
  std::vector< SharedMatrix > thread_mats(nthread);
  for (int i=0;i<nthread;i++){
    thread_mats[i]=SharedMatrix(new Matrix(nirrep_,nsopi_,nsopi_));
    thread_mats[i]->zero();
  }
  MintsHelper helper(basis_,RHF::options_,0);
  SharedMatrix aotoso = helper.petite_list(true)->aotoso();
  int *col_offset = new int [nirrep_];
  col_offset[0]=0;
  for (int h=1;h<nirrep_;h++){
    col_offset[h]=col_offset[h-1]+aotoso->coldim(h-1);
  }
  const int nao=basisset_->nao();
  const int nso=nso_;
  double **u = block_matrix(nao,nso);
  for(int h=0; h < nirrep_; h++){
    for(int j=0; j < aotoso->coldim(h); j++){
      for(int i=0; i < nao; i++){
        u[i][j+col_offset[h]] = aotoso->get(h, i, j);
      }
    }
  }
  #pragma omp parallel for schedule(auto)
    for (int i=0;i<nx_;i++){
      double xt=g_step_*i-centerx_;
      int myid=omp_get_thread_num();
      for (int j=0;j<ny_;j++){
        double yt=g_step_*j-centery_;
        for (int k=0;k<nz_;k++){
          double zt=g_step_*k-centerz_;
          int indx=i+nx_*j+nx_*ny_*k;
          double * phiao = new double [nao];
          double * phiso = new double [nso];
          basisset_->compute_phi(phiao,xt,yt,zt);
          C_DGEMV('t',nao,nso,1.0,&(u[0][0]),nso,&(phiao[0]),1,0.0,
                  &(phiso[0]),1);
         
          // calc sigsum here first 
          double sigsum=0.;
          for (int l=0;l<natom;l++){
            double dx=xt-geom.get(l,0);
            double dy=yt-geom.get(l,1);
            double dz=zt-geom.get(l,2);
            double r=sqrt(dx*dx+dy*dy+dz*dz);
            sigsum+=(Z[l]/r*erf(r/sigma_));  
          }

          for (int h=0;h<nirrep_;h++){
            for (int u=0;u<nsopi_[h];u++){
              for (int v=0;v<=u;v++){
                double val=phiso[u+col_offset[h]]*phiso[v+col_offset[h]]*sigsum;
                thread_mats[myid]->add(h,u,v,val);
              }
            }
          }
        delete [] phiao;
        delete [] phiso;
        }
      }
    }
  // Fill in rest of symmetric matrix
  for (int h=0;h<nirrep_;h++){
    for (int u=0;u<nsopi_[h];u++){
      for (int v=0;v<u;v++){
        for (int t=0;t<nthread;t++){
          V_sgao_->add(h,u,v,thread_mats[t]->get(h,u,v));
          V_sgao_->add(h,v,u,thread_mats[t]->get(h,u,v));
        }
      }
    }
  }

  // fill in diagonal part
  for (int h=0;h<nirrep_;h++){
    for (int u=0;u<nsopi_[h];u++){
      for (int t=0;t<nthread;t++){
        V_sgao_->add(h,u,u,thread_mats[t]->get(h,u,u));
      }
    }
  }
	V_sgao_->scale(pow(g_step_,3));
	V_sgao_->print_out();
  std::cout << "leaving v sgao calc" << std::endl;
  end=std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds=end-start;
  std::cout << "Calc_V_sgao took " << elapsed_seconds.count() << " seconds\n";
}


}} // End namespaces

