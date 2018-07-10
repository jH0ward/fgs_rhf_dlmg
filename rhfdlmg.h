#ifndef _rhfdlmg
#define _rhfdlmg

#include <vector>

#include "psi4/libmints/typedefs.h"
#include "psi4/libmints/oeprop.h"
#include "psi4/libscf_solver/rhf.h"
#include "psi4/libscf_solver/hf.h"
#include <psi4/libfock/cubature.h>


namespace psi{

namespace scf{

class RHFDLMG2B: public RHF{
public:
  RHFDLMG2B(SharedWavefunction ref_wfn,Options &options, std::shared_ptr<PSIO> psio);
  virtual ~RHFDLMG2B();
protected:
  SharedMatrix AO2USO_;
  SharedMatrix USO2AO_;
  std::shared_ptr<BlockOPoints> cube_;
  void calc_eps_symm();
  void density_block();
  void read_dens();
  std::string dens_file_;
  void read_ca();
  std::string ca_file_;
  void calc_sigsum();
  void compute_esp_ao();
  void compute_esp();
  virtual double compute_E();
  virtual void form_G();
  virtual void form_H();
  virtual double compute_energy();
  double calc_sg_pot();
  void calc_BC_vals();
	void calc_V_sol();
	void calc_V_sgao_P();
	void calc_V_eff();
	void calc_ndens_grid();
	void calc_dens_grid();
	double calc_sg_rep();
  SharedMatrix V_eff_;
  SharedMatrix V_sgao_;


    /// The number of atoms in the current molecule.
  int natom_;
  int nx_,ny_,nz_;
  int block_size_;
  std::vector<double> sigsum_;
  double g_step_,sigma_;

	// Vector to hold density at grid points
	std::vector<long double> lgmat_;
	std::vector<double> gmat_;
	std::vector<double> nmat_;
	std::vector<double> emat_;
	double centerx_,centery_,centerz_;
	int npts;
	double rho0_;//=0.00078;
  double beta2_;
  int dlmg_tol;
  double eps_inf_;
	std::vector<double> pot_sol_;
	std::vector<double> pot_vac_;
  std::vector<double> bound_;
	std::vector<double> eps_half_;
	std::vector<double> eps_full_;
  std::shared_ptr<BasisSet> basis_;
  int narray[3];
  double darray[3];
	
};

}} // Namespaces
#endif
