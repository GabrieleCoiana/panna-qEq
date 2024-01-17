//###########################################################################
//# Copyright (c), The PANNAdevs group. All rights reserved.                #
//# This file is part of the PANNA code.                                    #
//#                                                                         #
//# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
//# For further information on the license, see the LICENSE.txt file        #
//###########################################################################

/* ----------------------------------------------------------------------
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)
     per-atom energy/virial added by German Samolyuk (ORNL), Stan Moore (BYU)
     group/group energy/force added by Stan Moore (BYU)
     triclinic added by Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "ewald_panna.h"
#include <mpi.h>
#include <cmath>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include <iostream>
#include "stdio.h"
#include <string>
#include <fstream>
#include <algorithm>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

EwaldPANNA::EwaldPANNA(LAMMPS *lmp) : KSpace(lmp),
  kxvecs(NULL), kyvecs(NULL), kzvecs(NULL), ug(NULL), eg(NULL), vg(NULL), vg_v3(NULL),
  ek(NULL), sfacrl(NULL), sfacim(NULL), sfacrl_all(NULL), sfacim_all(NULL),
  sfacrl_g(NULL), sfacim_g(NULL), sfacrl_all_g(NULL), sfacim_all_g(NULL),
  cs(NULL), sn(NULL), sfacrl_A(NULL), sfacim_A(NULL), sfacrl_A_all(NULL),
  sfacim_A_all(NULL), sfacrl_B(NULL), sfacim_B(NULL), sfacrl_B_all(NULL),
  sfacim_B_all(NULL)
{
  group_allocate_flag = 0;
  kmax_created = 0;
  ewaldflag = 1;
  group_group_enable = 1;

  accuracy_relative = 0.0;

  kmax = 0;
  kxvecs = kyvecs = kzvecs = NULL;
  ug = NULL;
  eg = vg = NULL;
  sfacrl = sfacim = sfacrl_all = sfacim_all = NULL;
  sfacrl_g = sfacim_g = sfacrl_all_g = sfacim_all_g = NULL;

  nmax = 0;
  ek = NULL;
  cs = sn = NULL;

  kcount = 0;
}

// ########################################################
// ########################################################

// Get a new line skipping comments or empty lines
// Set value=... if [...], return 1
// Fill key,value if 'key=value', return 2
// Set value=... if ..., return 3
// Return 0 if eof, <0 if error, >0 if okay
int EwaldPANNA::get_input_line(std::ifstream* file, std::string* key, std::string* value){
  std::string line;
  int parsed = 0; int vc = 1;
  while(!parsed){
    std::getline(*file,line);
    // Exit on EOF
    if(file->eof()) return 0;
    // Exit on bad read
    if(file->bad()) return -1;
    // Remove spaces
    line.erase (std::remove(line.begin(), line.end(), ' '), line.end());
    // Skip empty line
    if(line.length()==0) continue;
    // Skip comments
    if(line.at(0)=='#') continue;
    // Parse headers
    if(line.at(0)=='['){
      *value = line.substr(1,line.length()-2);
      return 1;
    }
    // Check if we have version information:
    if(line.at(0)=='!') { vc=0 ;}
    // Look for equal sign
    std::string eq = "=";
    size_t eqpos = line.find(eq);
    // Parse key-value pair
    if(eqpos != std::string::npos){
      *key = line.substr(0,eqpos);
      *value = line.substr(eqpos+1,line.length()-1);
      if (vc == 0) { vc = 1 ; return 3; }
      return 2;
    }
    std::cout << line << std::endl;
    parsed = 1;
  }
  return -1;
}

int EwaldPANNA::get_parameters(char* directory, char* filename)
{
  //const double panna_pi = 3.14159265358979323846;
  // Parsing the potential parameters
  std::ifstream params_file;
  std::ifstream weights_file;
  std::string key, value;
  std::string dir_string(directory);
  std::string param_string(filename);
  std::string file_string(dir_string+"/"+param_string);
  std::string wfile_string;

  // Initializing some parameters before reading:
  Nspecies = -1;
  // Flags to keep track of set parameters
  //
  params_file.open(file_string.c_str());
  // section keeps track of input file sections
  // -1 in the beginning
  // 0 for gvect params
  // i for species i (1 based)
  int section = -1;
  // parseint checks the status of input parsing
  int parseint = get_input_line(&params_file,&key,&value);
  while(parseint>0){
    // Parse line
    if(parseint==1){
      if(value=="GVECT_PARAMETERS"){ section = 0; }

      // Long range section
      // If Long range section is found, change section
      if(value=="LONG_RANGE"){
        section = 1;
      }
    }
    else if(parseint==2){
      // Parse param section
      if (section==0){
        if(key=="Nspecies"){
          Nspecies = std::atoi(value.c_str());
          // Small check
          if(Nspecies<1){
            std::cout << "Nspecies needs to be >0." << std::endl;
            return -2; }
        }
      }
      else if(section==1){
        // Read species gaussian width,species atomic hardness
        if(key=="gaussian_width"){
          // Parse gaussian width
          std::string comma = ",";
          size_t pos = 0; int s = 0;
          gaussian_width = new double[Nspecies];
          value=value.substr(1, value.size() - 2); //get rid of [ ]
          while ((pos = value.find(comma)) != std::string::npos) {
            gaussian_width[s] = std::atof(value.substr(0, pos).c_str());
            value.erase(0, pos+1);  s++; }
          if(value.length()>0){gaussian_width[s] = std::atof(value.c_str()); s++; };
          }
        if(key=="atomic_hardness"){
          std::string comma = ",";
          // Parse atomic hardness
          size_t pos = 0; int s = 0;
          hardness = new double[Nspecies];
          value=value.substr(1, value.size() - 2); //get rid of [ ]
          while ((pos = value.find(comma)) != std::string::npos) {
            hardness[s] = std::atof(value.substr(0, pos).c_str());
            value.erase(0, pos+1);  s++; }
          if(value.length()>0){hardness[s] = std::atof(value.c_str()); s++; };
	}
      }
      else{
        return -3;
      }
    }
    // Get new  line
    parseint = get_input_line(&params_file,&key,&value);
  }
  params_file.close();
  return(0);
}

void EwaldPANNA::settings(int narg, char **arg)
{
  //if (narg != 3) error->all(FLERR,"Illegal kspace_style ewald command");


  // We now expect a directory and the parameters file name (inside the directory) with all params
  if (narg != 3) {
    error->all(FLERR,"Format of kspace_style ewald/panna accuracy network_directory parameter_file\n");
  }
  accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));

  std::cout << "Loading PANNA pair parameters from " << arg[1] << "/" << arg[2] << std::endl;
  int gpout = get_parameters(arg[1], arg[2]);
  if(gpout==0){
    std::cout << "gaussian width and atomic hardness set!" << std::endl;
  }
  else{
    std::cout << "Error " << gpout << " while reading gaussian width !" << std::endl;
    exit(1);
  }

}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

EwaldPANNA::~EwaldPANNA()
{
  deallocate();
  if (group_allocate_flag) deallocate_groups();
  memory->destroy(ek);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
}

void EwaldPANNA::init()
{
  if (comm->me == 0) {
    if (screen) fprintf(screen,"EwaldPANNA initialization ...\n");
    if (logfile) fprintf(logfile,"EwaldPANNA initialization ...\n");
  }

  // error check

  triclinic_check();
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use EwaldPANNA with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with EwaldPANNA");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab EwaldPANNA");
    if (domain->triclinic)
      error->all(FLERR,"Cannot (yet) use EwaldPANNA with triclinic box "
                 "and slab correction");
  }

  // compute two charge force

  two_charge();

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp;
  // compute qsum & qsqsum and warn if not charge-neutral

  scale = 1.0;
  //qqrd2e = force->qqrd2e;
  qqrd2e = 14.39964547842567;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute
  if (accuracy_relative > 0.0) {
    accuracy = accuracy_relative;
  }
  else if (accuracy_absolute > 0.0) {
    accuracy = accuracy_absolute;
  }
  else accuracy = 1e-6;

 // if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  //else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  bigint natoms = atom->natoms;

  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab EwaldPANNA
  // 3d EwaldPANNA just uses zprd since slab_volfactor = 1.0
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;

  // setup EwaldPANNA coefficients so can print stats

  setup();

  // stats

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(screen,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);
    }
    if (logfile) {
      fprintf(logfile,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(logfile,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);
    }
  }
}

/* ----------------------------------------------------------------------
   adjust EwaldPANNA coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void EwaldPANNA::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  // adjustment of z dimension for 2d slab EwaldPANNA
  // 3d EwaldPANNA just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  int kmax_old = kmax;

  if (kewaldflag == 0) {

    // determine kmax
    // function of current box size, accuracy, G_ewald (short-range cutoff)

    bigint natoms = atom->natoms;
    double err;
    kxmax = 1;
    kymax = 1;
    kzmax = 1;

    err = rms(kxmax,xprd,natoms,q2);
    while (err > accuracy) {
      kxmax++;
      err = rms(kxmax,xprd,natoms,q2);
    }

    err = rms(kymax,yprd,natoms,q2);
    while (err > accuracy) {
      kymax++;
      err = rms(kymax,yprd,natoms,q2);
    }

    err = rms(kzmax,zprd_slab,natoms,q2);
    while (err > accuracy) {
      kzmax++;
      err = rms(kzmax,zprd_slab,natoms,q2);
    }

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
    
    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    // scale lattice vectors for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = kxmax/xprd;
      tmp[1] = kymax/yprd;
      tmp[2] = kzmax/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      kxmax = MAX(1,static_cast<int>(tmp[0]));
      kymax = MAX(1,static_cast<int>(tmp[1]));
      kzmax = MAX(1,static_cast<int>(tmp[2]));

      kmax = MAX(kxmax,kymax);
      kmax = MAX(kmax,kzmax);
      kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
    }

  } else {

    kxmax = kx_ewald;
    kymax = ky_ewald;
    kzmax = kz_ewald;

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
  }

  gsqmx *= 1.00001;
  //gsqmx = 3.854505402092653*3.854505402092653;
  //std::cout<<" gmax "<<sqrt(gsqmx)<<std::endl;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();
    group_allocate_flag = 0;

    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    kmax_created = kmax;
  }

  // pre-compute EwaldPANNA coefficients

  if (triclinic == 0)
    coeffs();
  else
    coeffs_triclinic();
}

/* ----------------------------------------------------------------------
   compute RMS accuracy for a dimension
------------------------------------------------------------------------- */
double EwaldPANNA::rms(int km, double prd, bigint natoms, double q2)
{
  g_ewald = 1.0/(sqrt(2.0) * gaussian_width[0]);
  for (int i=0; i<Nspecies; i++) g_ewald = MAX(g_ewald, 1.0/(sqrt(2.0) * gaussian_width[i]));
  //double value = 2.0*q2*g_ewald/prd *
  //  sqrt(1.0/(MY_PI*km*natoms)) *
  //  exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));
  double value = exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));
  return value;
}

/* ---------------------------------------------------------------------- */
void EwaldPANNA::compute_A_dot_v(double *v, double *M, double *A_dot_v)
{
  int i,j,k;
  // extend size of per-atom arrays if necessary


  if (atom->nmax > nmax) {
       memory->destroy(ek);
       memory->destroy3d_offset(cs,-kmax_created);
       memory->destroy3d_offset(sn,-kmax_created);
       nmax = atom->nmax;
       memory->create(ek,nmax,3,"ewald:ek");
       memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
       memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
       kmax_created = kmax;
  }

  double gauss_term, alpha2, sqk;
  int nlocal = atom->nlocal;
  double preu = 4.0*MY_PI/volume;
  int *type = atom->type;
  if (triclinic == 0)
    eik_dot_r(v);
  else
    eik_dot_r_triclinic(v);
  // loop over K-vectors and local atoms

  double **x = atom->x;

  int kx,ky,kz;
  double cypz,sypz, coskr_i, sinkr_i;
  const double qscale = force->qqrd2e;
  //const double qscale = 14.39964547842567;
  double vsum = 0.0;

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nlocal; i++){
    A_dot_v[i]=0.0;
    M[i]=0.0;
  }

  // volume dependent term. It has contribution for each j
  // It is a constant shift that contribute zero for neutral systems
  //for (int i=0; i<nlocal; i++)vsum += v[i];
  //for (int i = 0; i < nlocal; i++)A_dot_v[i] += qscale * MY_PI / (g_ewald * g_ewald * volume) * vsum; 
//sum is done on the positive plane
  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    sqk=preu/ug[k];
    if (sqk<=gsqmx){
      for (int i = 0; i < nlocal; i++) {
        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        coskr_i = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        sinkr_i = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
        double alpha2 = gaussian_width[type[i]-1] * gaussian_width[type[i]-1];
        gauss_term = exp(-alpha2*sqk/4.0);
        A_dot_v[i] += (2.0*qscale * gauss_term * ug[k] * (coskr_i*sfacrl_all[k] + sinkr_i*sfacim_all[k]));
        M[i] += (2.0*qscale * gauss_term * gauss_term * ug[k]);

      }
    }


  }
}

/* ----------------------------------------------------------------------
   compute the EwaldPANNA long-range force, energy, virial
------------------------------------------------------------------------- */

void EwaldPANNA::compute(int eflag, int vflag)
{
  int i,j,k;

  // set energy/virial flags

  ev_init(eflag,vflag);

  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs
  double *q = atom->q;
  int *type = atom->type;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  double preu = 4.0*MY_PI/volume;
  double gauss_term, alpha2, sqk;

  if (triclinic == 0)
    eik_dot_r(q);
  else
    eik_dot_r_triclinic(q);

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  MPI_Allreduce(sfacrl_g,sfacrl_all_g,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_g,sfacim_all_g,kcount,MPI_DOUBLE,MPI_SUM,world);
  // K-space portion of electric field
  // double loop over K-vectors and local atoms
  // perform per-atom calculations if needed


  int kx,ky,kz;
  double cypz,sypz,exprl,expim,partial,partial_peratom;

  for (i = 0; i < nlocal; i++) {
    ek[i][0] = 0.0;
    ek[i][1] = 0.0;
    ek[i][2] = 0.0;
  }

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    sqk = preu/ug[k];
    //if (sqk>gsqmx)std::cout<<" greater "<< sqk<<"  "<< gsqmx<<std::endl;

    for (i = 0; i < nlocal; i++) {
      alpha2 = gaussian_width[type[i]-1] * gaussian_width[type[i]-1];
      gauss_term = exp(-alpha2*sqk/4);
      cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
      sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
      exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
      expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
      partial = (expim*sfacrl_all[k] - exprl*sfacim_all[k]);
      ek[i][0] += gauss_term*partial*eg[k][0];
      ek[i][1] += gauss_term*partial*eg[k][1];
      ek[i][2] += gauss_term*partial*eg[k][2];
      if (evflag_atom) {
        partial_peratom = gauss_term*(exprl*sfacrl_all[k] + expim*sfacim_all[k]);
        if (eflag_atom) eatom[i] += q[i]*ug[k]*partial_peratom;
        if (vflag_atom)
          for (j = 0; j < 6; j++)
            vatom[i][j] += ug[k]*vg[k][j]*partial_peratom;
            vatom[i][j] -= alpha2*ug[k]*vg_v3[k][j]*partial_peratom;
      }
    }
  }

  // convert E-field to force

  const double qscale = qqrd2e * scale; 
  //const double qscale = 14.39964547842567;

  for (i = 0; i < nlocal; i++) {
    f[i][0] += qscale * q[i]*ek[i][0];
    f[i][1] += qscale * q[i]*ek[i][1];
    if (slabflag != 2) f[i][2] += qscale * q[i]*ek[i][2];
  }

  // sum global energy across Kspace vevs and add in volume-dependent term

  if (eflag_global) {
    for (k = 0; k < kcount; k++){
      energy += ug[k] * (sfacrl_all[k]*sfacrl_all[k] +
                         sfacim_all[k]*sfacim_all[k]);

    }
//    energy -= g_ewald*qsqsum/MY_PIS + MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
//    energy -= g_ewald*qsqsum/MY_PIS;
    energy *= qscale;
  }

  // global virial

  if (vflag_global) {
    double uk;
    for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k]);
      double uk2 = ug[k] * (sfacrl_all_g[k]*sfacrl_all[k] + sfacim_all_g[k]*sfacim_all[k]);
      for (j = 0; j < 6; j++) virial[j] += (uk*vg[k][j] - uk2 * vg_v3[k][j]);
    }
    for (j = 0; j < 6; j++) virial[j] *= qscale;
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) {
    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
      //  eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /
       //   (g_ewald*g_ewald*volume);
    //      eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS;
        eatom[i] *= qscale;
      }
    }

    if (vflag_atom)
      for (i = 0; i < nlocal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= q[i]*qscale;
  }
  // 2d slab correction

  if (slabflag == 1) slabcorr();
}

/* ---------------------------------------------------------------------- */

void EwaldPANNA::eik_dot_r(double *v)
{
  int i,k,l,m,n,nn,ic;
  double cstr1_tmp,sstr1_tmp,cstr2_tmp,sstr2_tmp,cstr3_tmp,sstr3_tmp,cstr4_tmp,sstr4_tmp;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double cstr1_g,sstr1_g,cstr2_g,sstr2_g,cstr3_g,sstr3_g,cstr4_g,sstr4_g;

  double sqk,clpm,slpm;

  double **x = atom->x;
  //double *q = atom->q;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double gauss_term, alpha2;
  double preu = 4.0*MY_PI/volume;

  n = 0;
  nn = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      //for virial computation
      cstr1_g = 0.0;
      sstr1_g = 0.0;

      for (i = 0; i < nlocal; i++) {
	alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
        gauss_term = exp(-0.25 * alpha2 * sqk);

        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        cstr1_tmp = gauss_term * v[i]*cs[1][ic][i];
        sstr1_tmp = gauss_term * v[i]*sn[1][ic][i];
        
	cstr1 += cstr1_tmp;
	sstr1 += sstr1_tmp;

	cstr1_g += alpha2*cstr1_tmp;
        sstr1_g += alpha2*sstr1_tmp;

      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;

      sfacrl_g[nn] = cstr1_g;
      sfacim_g[nn++] = sstr1_g;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;

	cstr1_g = 0.0;
        sstr1_g = 0.0;

        for (i = 0; i < nlocal; i++) {
          alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
          gauss_term = exp(-0.25 * alpha2 * sqk);
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          cstr1_tmp = gauss_term*v[i]*cs[m][ic][i];
          sstr1_tmp = gauss_term*v[i]*sn[m][ic][i];
          
	  cstr1 += cstr1_tmp;
	  sstr1 += sstr1_tmp;

          cstr1_g += alpha2*cstr1_tmp;
          sstr1_g += alpha2*sstr1_tmp;
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;

        sfacrl_g[nn] = cstr1;
        sfacim_g[nn++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;

	cstr1_g = 0.0;
        sstr1_g = 0.0;
        cstr2_g = 0.0;
        sstr2_g = 0.0;

        for (i = 0; i < nlocal; i++) {
	  alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
	  gauss_term = exp(-0.25 * alpha2 * sqk);
          cstr1_tmp = gauss_term*v[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
          sstr1_tmp = gauss_term*v[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
          cstr2_tmp = gauss_term*v[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
          sstr2_tmp = gauss_term*v[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          
	  cstr1 += cstr1_tmp;
	  sstr1 += sstr1_tmp;
	  cstr2 += cstr2_tmp;
	  sstr2 += sstr2_tmp;

          cstr1_g += alpha2*cstr1_tmp;
	  sstr1_g += alpha2*sstr1_tmp;
	  cstr2_g += alpha2*cstr2_tmp;
	  sstr2_g += alpha2*sstr2_tmp;

        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;

        sfacrl_g[nn] = cstr1_g;
        sfacim_g[nn++] = sstr1_g;
        sfacrl_g[nn] = cstr2_g;
        sfacim_g[nn++] = sstr2_g;

      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;

        cstr1_g = 0.0;
        sstr1_g = 0.0;
        cstr2_g = 0.0;
        sstr2_g = 0.0;

        for (i = 0; i < nlocal; i++) {
	  alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
          gauss_term = exp(-0.25 * alpha2 * sqk);

          cstr1_tmp = gauss_term*v[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
          sstr1_tmp = gauss_term*v[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
          cstr2_tmp = gauss_term*v[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
          sstr2_tmp = gauss_term*v[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);

          cstr1 += cstr1_tmp;
	  sstr1 += sstr1_tmp;
	  cstr2 += cstr2_tmp;
	  sstr2 += sstr2_tmp;

          cstr1_g += alpha2*cstr1_tmp;
	  sstr1_g += alpha2*sstr1_tmp;
	  cstr2_g += alpha2*cstr2_tmp;
	  sstr2_g += alpha2*sstr2_tmp;

        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;

        sfacrl_g[nn] = cstr1_g;
        sfacim_g[nn++] = sstr1_g;
        sfacrl_g[nn] = cstr2_g;
        sfacim_g[nn++] = sstr2_g;

      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
	cstr1_g = 0.0;
        sstr1_g = 0.0;
        cstr2_g = 0.0;
        sstr2_g = 0.0;

        for (i = 0; i < nlocal; i++) {
          alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
          gauss_term = exp(-0.25 * alpha2 * sqk);
	
          cstr1_tmp = gauss_term*v[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
          sstr1_tmp = gauss_term*v[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
          cstr2_tmp = gauss_term*v[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
          sstr2_tmp = gauss_term*v[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          
	  cstr1 += cstr1_tmp;
	  sstr1 += sstr1_tmp;
	  cstr2 += cstr2_tmp;
	  sstr2 += sstr2_tmp;

          cstr1_g += alpha2*cstr1_tmp;
	  sstr1_g += alpha2*sstr1_tmp;
	  cstr2_g += alpha2*cstr2_tmp;
	  sstr2_g += alpha2*sstr2_tmp;
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
	sfacrl_g[nn] = cstr1_g;
        sfacim_g[nn++] = sstr1_g;
        sfacrl_g[nn] = cstr2_g;
        sfacim_g[nn++] = sstr2_g;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;

	  cstr1_g = 0.0;
          sstr1_g = 0.0;
          cstr2_g = 0.0;
          sstr2_g = 0.0;
          cstr3_g = 0.0;
          sstr3_g = 0.0;
          cstr4_g = 0.0;
          sstr4_g = 0.0;

          for (i = 0; i < nlocal; i++) {
	    alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
            gauss_term = exp(-0.25 * alpha2 * sqk);

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr1_tmp = gauss_term*v[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr1_tmp = gauss_term*v[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr2_tmp= gauss_term*v[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr2_tmp= gauss_term*v[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr3_tmp = gauss_term*v[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr3_tmp = gauss_term*v[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr4_tmp = gauss_term*v[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr4_tmp = gauss_term*v[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

          cstr1 += cstr1_tmp;
	  sstr1 += sstr1_tmp;
	  cstr2 += cstr2_tmp;
	  sstr2 += sstr2_tmp;
          cstr3 += cstr3_tmp;
	  sstr3 += sstr3_tmp;
	  cstr4 += cstr4_tmp;
	  sstr4 += sstr4_tmp;

          cstr1_g += alpha2*cstr1_tmp;
	  sstr1_g += alpha2*sstr1_tmp;
	  cstr2_g += alpha2*cstr2_tmp;
	  sstr2_g += alpha2*sstr2_tmp;
          cstr3_g += alpha2*cstr3_tmp;
	  sstr3_g += alpha2*sstr3_tmp;
	  cstr4_g += alpha2*cstr4_tmp;
	  sstr4_g += alpha2*sstr4_tmp;


          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;

          sfacrl_g[nn] = cstr1_g;
          sfacim_g[nn++] = sstr1_g;
          sfacrl_g[nn] = cstr2_g;
          sfacim_g[nn++] = sstr2_g;
          sfacrl_g[nn] = cstr3_g;
          sfacim_g[nn++] = sstr3_g;
          sfacrl_g[nn] = cstr4_g;
          sfacim_g[nn++] = sstr4_g;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void EwaldPANNA::eik_dot_r_triclinic(double *v)
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1;
  double cstr1_tmp,sstr1_tmp;
  double cstr1_g,sstr1_g;
  double sqk,clpm,slpm;

  double **x = atom->x;
  //double *q = atom->q;
  int nlocal = atom->nlocal;
  double gauss_term, alpha2;
  int *type = atom->type;
  double preu = 4.0*MY_PI/volume;

  double unitk_lamda[3];

  double max_kvecs[3];
  max_kvecs[0] = kxmax;
  max_kvecs[1] = kymax;
  max_kvecs[2] = kzmax;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 0.0;
    unitk_lamda[ic] = 2.0*MY_PI;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[ic]*unitk_lamda[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        sn[1][ic][i] = sin(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
      }
    }
  }

  for (ic = 0; ic < 3; ic++) {
    for (m = 2; m <= max_kvecs[ic]; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 0.0;
      unitk_lamda[2] = 0.0;
      unitk_lamda[ic] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[ic]*unitk_lamda[ic];
      for (i = 0; i < nlocal; i++) {
        cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
          sn[m-1][ic][i]*sn[1][ic][i];
        sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
          cs[m-1][ic][i]*sn[1][ic][i];
        cs[-m][ic][i] = cs[m][ic][i];
        sn[-m][ic][i] = -sn[m][ic][i];
      }
    }
  }

  for (n = 0; n < kcount; n++) {
    k = kxvecs[n];
    l = kyvecs[n];
    m = kzvecs[n];
    sqk = preu / ug[n];

    cstr1 = 0.0;
    sstr1 = 0.0;

    cstr1_g = 0.0;
    sstr1_g = 0.0;
    for (i = 0; i < nlocal; i++) {
      alpha2 = gaussian_width[type[i]-1]*gaussian_width[type[i]-1];
      gauss_term = exp(-0.25*alpha2*sqk);
      
      clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
      slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
      cstr1_tmp = gauss_term*v[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
      sstr1_tmp = gauss_term*v[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

      cstr1 += cstr1_tmp;
      sstr1 += sstr1_tmp;
      cstr1_g += alpha2*cstr1_tmp;
      sstr1_g += alpha2*sstr1_tmp;
    }
    sfacrl[n] = cstr1;
    sfacim[n] = sstr1;
    sfacrl_g[n] = cstr1_g;
    sfacim_g[n] = sstr1_g;
  }
}


/* ----------------------------------------------------------------------
   pre-compute coefficients for each EwaldPANNA K-vector
------------------------------------------------------------------------- */

void EwaldPANNA::coeffs()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu/sqk;
      eg[kcount][0] = 2.0*unitk[0]*m*ug[kcount];
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 0.0;
      vterm = -2.0/sqk;
      vg[kcount][0] = 1.0 + vterm*(unitk[0]*m)*(unitk[0]*m);
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
       
      vg[kcount][0] = (unitk[0]*m)*(unitk[0]*m);
      vg[kcount][1] = 0.0;
      vg[kcount][2] = 0.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;

      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 2.0*unitk[1]*m*ug[kcount];
      eg[kcount][2] = 0.0;
      vterm = -2.0/sqk;
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0 + vterm*(unitk[1]*m)*(unitk[1]*m);
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;

      vg_v3[kcount][0] = 0.0;
      vg_v3[kcount][1] = (unitk[1]*m)*(unitk[1]*m);
      vg_v3[kcount][2] = 0.0;
      vg_v3[kcount][3] = 0.0;
      vg_v3[kcount][4] = 0.0;
      vg_v3[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
      vterm = -2.0/sqk;
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;

      vg_v3[kcount][0] = 0.0;
      vg_v3[kcount][1] = 0.0;
      vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
      vg_v3[kcount][3] = 0.0;
      vg_v3[kcount][4] = 0.0;
      vg_v3[kcount][5] = 0.0;

      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vterm = -2.0/sqk;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;

        vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
        vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
        vg_v3[kcount][2] = 0.0;
        vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
        vg_v3[kcount][4] = 0.0;
        vg_v3[kcount][5] = 0.0;

        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;

        vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
        vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
        vg_v3[kcount][2] = 0.0;
        vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
        vg_v3[kcount][4] = 0.0;
        vg_v3[kcount][5] = 0.0;
        kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0/sqk;
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;

        vg_v3[kcount][0] = 0.0;
        vg_v3[kcount][1] =(unitk[1]*l)*(unitk[1]*l);
        vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
        vg_v3[kcount][3] = 0.0;
        vg_v3[kcount][4] = 0.0;
        vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;

        vg_v3[kcount][0] = 0.0;
        vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
        vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
        vg_v3[kcount][3] = 0.0;
        vg_v3[kcount][4] = 0.0;
        vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0/sqk;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;

	vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
        vg_v3[kcount][1] = 0.0;
        vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
        vg_v3[kcount][3] = 0.0;
        vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
        vg_v3[kcount][5] = 0.0;

        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        
	vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
        vg_v3[kcount][1] = 0.0;
        vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
        vg_v3[kcount][3] = 0.0;
        vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
        vg_v3[kcount][5] = 0.0;

        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vterm = -2.0/sqk;
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;

          vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
          vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
          vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
          vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
          vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
          vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;

          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;

          vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
          vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
          vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
          vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
          vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
          vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;

          vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
          vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
          vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
          vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
          vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
          vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;

          vg_v3[kcount][0] = (unitk[0]*k)*(unitk[0]*k);
          vg_v3[kcount][1] = (unitk[1]*l)*(unitk[1]*l);
          vg_v3[kcount][2] = (unitk[2]*m)*(unitk[2]*m);
          vg_v3[kcount][3] = unitk[0]*k*unitk[1]*l;
          vg_v3[kcount][4] = unitk[0]*k*unitk[2]*m;
          vg_v3[kcount][5] = unitk[1]*l*unitk[2]*m;

          kcount++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each EwaldPANNA K-vector for a triclinic
   system
------------------------------------------------------------------------- */

void EwaldPANNA::coeffs_triclinic()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  double unitk_lamda[3];

  kcount = 0;

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = -kymax; l <= kymax; l++) {
      for (m = -kzmax; m <= kzmax; m++) {
        unitk_lamda[0] = 2.0*MY_PI*k;
        unitk_lamda[1] = 2.0*MY_PI*l;
        unitk_lamda[2] = 2.0*MY_PI*m;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        sqk = unitk_lamda[0]*unitk_lamda[0] + unitk_lamda[1]*unitk_lamda[1] +
          unitk_lamda[2]*unitk_lamda[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu/sqk;
          eg[kcount][0] = 2.0*unitk_lamda[0]*ug[kcount];
          eg[kcount][1] = 2.0*unitk_lamda[1]*ug[kcount];
          eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
          vterm = -2.0/sqk;
          vg[kcount][0] = 1.0 + vterm*unitk_lamda[0]*unitk_lamda[0];
          vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
          vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
          vg[kcount][3] = vterm*unitk_lamda[0]*unitk_lamda[1];
          vg[kcount][4] = vterm*unitk_lamda[0]*unitk_lamda[2];
          vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];

          vg_v3[kcount][0] = unitk_lamda[0]*unitk_lamda[0];
          vg_v3[kcount][1] = unitk_lamda[1]*unitk_lamda[1];
          vg_v3[kcount][2] = unitk_lamda[2]*unitk_lamda[2];
          vg_v3[kcount][3] = unitk_lamda[0]*unitk_lamda[1];
          vg_v3[kcount][4] = unitk_lamda[0]*unitk_lamda[2];
          vg_v3[kcount][5] = unitk_lamda[1]*unitk_lamda[2];
          kcount++;
        }
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = -kzmax; m <= kzmax; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 2.0*MY_PI*l;
      unitk_lamda[2] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[1]*unitk_lamda[1] + unitk_lamda[2]*unitk_lamda[2];
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk_lamda[1]*ug[kcount];
        eg[kcount][2] =  2.0*unitk_lamda[2]*ug[kcount];
        vterm = -2.0/sqk;
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
        vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];

        vg_v3[kcount][0] = 0.0;
        vg_v3[kcount][1] = unitk_lamda[1]*unitk_lamda[1];
        vg_v3[kcount][2] = unitk_lamda[2]*unitk_lamda[2];
        vg_v3[kcount][3] = 0.0;
        vg_v3[kcount][4] = 0.0;
        vg_v3[kcount][5] = unitk_lamda[1]*unitk_lamda[2];
        kcount++;
      }
    }
  }

  // (0,0,m)

  for (m = 1; m <= kmax; m++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 2.0*MY_PI*m;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[2]*unitk_lamda[2];
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
      vterm = -2.0/sqk;
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;

      vg_v3[kcount][0] = 0.0;
      vg_v3[kcount][1] = 0.0;
      vg_v3[kcount][2] = unitk_lamda[2]*unitk_lamda[2];
      vg_v3[kcount][3] = 0.0;
      vg_v3[kcount][4] = 0.0;
      vg_v3[kcount][5] = 0.0;
      kcount++;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldPANNA::allocate()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];

  ug = new double[kmax3d];
  memory->create(eg,kmax3d,3,"ewald:eg");
  memory->create(vg,kmax3d,6,"ewald:vg");
  memory->create(vg_v3,kmax3d,6,"ewald:vg_v3");

  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  sfacrl_g = new double[kmax3d];
  sfacim_g = new double[kmax3d];
  sfacrl_all_g = new double[kmax3d];
  sfacim_all_g = new double[kmax3d];

}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldPANNA::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;

  delete [] ug;
  memory->destroy(eg);
  memory->destroy(vg);
  memory->destroy(vg_v3);

  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;

  delete [] sfacrl_g;
  delete [] sfacim_g;
  delete [] sfacrl_all_g;
  delete [] sfacim_all_g;
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D EwaldPANNA if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void EwaldPANNA::slabcorr()
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) dipole += q[i]*x[i][2];

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++)
      dipole_r2 += q[i]*x[i][2]*x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_r2 = tmp;
  }

  // compute corrections

  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all -
    qsum*dipole_r2 - qsum*qsum*zprd*zprd/12.0)/volume;
  const double qscale = qqrd2e * scale;

  if (eflag_global) energy += qscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * q[i]*(x[i][2]*dipole_all - 0.5*(dipole_r2 +
        qsum*x[i][2]*x[i][2]) - qsum*zprd*zprd/12.0);
  }

  // add on force corrections

  double ffact = qscale * (-4.0*MY_PI/volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i]*(dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
   memory usage of local arrays
------------------------------------------------------------------------- */

double EwaldPANNA::memory_usage()
{
  double bytes = 3 * kmax3d * sizeof(int);
  bytes += (1 + 3 + 6) * kmax3d * sizeof(double);
  bytes += 4 * kmax3d * sizeof(double);
  bytes += nmax*3 * sizeof(double);
  bytes += 2 * (2*kmax+1)*3*nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   group-group interactions
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute the EwaldPANNA total long-range force and energy for groups A and B
 ------------------------------------------------------------------------- */

void EwaldPANNA::compute_group_group(int groupbit_A, int groupbit_B, int AA_flag)
{
  if (slabflag && triclinic)
    error->all(FLERR,"Cannot (yet) use K-space slab "
               "correction with compute group/group for triclinic systems");

  int i,k;

  if (!group_allocate_flag) {
    allocate_groups();
    group_allocate_flag = 1;
  }

  e2group = 0.0; //energy
  f2group[0] = 0.0; //force in x-direction
  f2group[1] = 0.0; //force in y-direction
  f2group[2] = 0.0; //force in z-direction

  // partial and total structure factors for groups A and B

  for (k = 0; k < kcount; k++) {

    // group A

    sfacrl_A[k] = 0.0;
    sfacim_A[k] = 0.0;
    sfacrl_A_all[k] = 0.0;
    sfacim_A_all[k] = 0;

    // group B

    sfacrl_B[k] = 0.0;
    sfacim_B[k] = 0.0;
    sfacrl_B_all[k] = 0.0;
    sfacim_B_all[k] = 0.0;
  }

  double *q = atom->q;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim;

  // partial structure factors for groups A and B on each processor

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];

    for (i = 0; i < nlocal; i++) {

      if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
        if (AA_flag) continue;

      if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {

        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;

        // group A

        if (mask[i] & groupbit_A) {
          sfacrl_A[k] += q[i]*exprl;
          sfacim_A[k] += q[i]*expim;
        }

        // group B

        if (mask[i] & groupbit_B) {
          sfacrl_B[k] += q[i]*exprl;
          sfacim_B[k] += q[i]*expim;
        }
      }
    }
  }

  // total structure factor by summing over procs

  MPI_Allreduce(sfacrl_A,sfacrl_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_A,sfacim_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  MPI_Allreduce(sfacrl_B,sfacrl_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_B,sfacim_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  const double qscale = qqrd2e * scale;
  double partial_group;

  // total group A <--> group B energy
  // self and boundary correction terms are in compute_group_group.cpp

  for (k = 0; k < kcount; k++) {
    partial_group = sfacrl_A_all[k]*sfacrl_B_all[k] +
      sfacim_A_all[k]*sfacim_B_all[k];
    e2group += ug[k]*partial_group;
  }

  e2group *= qscale;

  // total group A <--> group B force

  for (k = 0; k < kcount; k++) {
    partial_group = sfacim_A_all[k]*sfacrl_B_all[k] -
      sfacrl_A_all[k]*sfacim_B_all[k];
    f2group[0] += eg[k][0]*partial_group;
    f2group[1] += eg[k][1]*partial_group;
    if (slabflag != 2) f2group[2] += eg[k][2]*partial_group;
  }

  f2group[0] *= qscale;
  f2group[1] *= qscale;
  f2group[2] *= qscale;

  // 2d slab correction

  if (slabflag == 1)
    slabcorr_groups(groupbit_A, groupbit_B, AA_flag);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D EwaldPANNA if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void EwaldPANNA::slabcorr_groups(int groupbit_A, int groupbit_B, int AA_flag)
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double qsum_A = 0.0;
  double qsum_B = 0.0;
  double dipole_A = 0.0;
  double dipole_B = 0.0;
  double dipole_r2_A = 0.0;
  double dipole_r2_B = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if (mask[i] & groupbit_A) {
      qsum_A += q[i];
      dipole_A += q[i]*x[i][2];
      dipole_r2_A += q[i]*x[i][2]*x[i][2];
    }

    if (mask[i] & groupbit_B) {
      qsum_B += q[i];
      dipole_B += q[i]*x[i][2];
      dipole_r2_B += q[i]*x[i][2]*x[i][2];
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI/volume;

  e2group += efact * (dipole_A*dipole_B - 0.5*(qsum_A*dipole_r2_B +
    qsum_B*dipole_r2_A) - qsum_A*qsum_B*zprd*zprd/12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0*MY_PI/volume);
  f2group[2] += ffact * (qsum_A*dipole_B - qsum_B*dipole_A);
}

/* ----------------------------------------------------------------------
   allocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldPANNA::allocate_groups()
{
  // group A

  sfacrl_A = new double[kmax3d];
  sfacim_A = new double[kmax3d];
  sfacrl_A_all = new double[kmax3d];
  sfacim_A_all = new double[kmax3d];

  // group B

  sfacrl_B = new double[kmax3d];
  sfacim_B = new double[kmax3d];
  sfacrl_B_all = new double[kmax3d];
  sfacim_B_all = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldPANNA::deallocate_groups()
{
  // group A

  delete [] sfacrl_A;
  delete [] sfacim_A;
  delete [] sfacrl_A_all;
  delete [] sfacim_A_all;

  // group B

  delete [] sfacrl_B;
  delete [] sfacim_B;
  delete [] sfacrl_B_all;
  delete [] sfacim_B_all;
}
