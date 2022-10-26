//###########################################################################
//# Copyright (c), The PANNAdevs group. All rights reserved.                #
//# This file is part of the PANNA code.                                    #
//#                                                                         #
//# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
//# For further information on the license, see the LICENSE.txt file        #
//###########################################################################

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_panna.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */


// ########################################################
//                       Constructor
// ########################################################
//

PairPANNA::PairPANNA(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  // feenableexcept(FE_INVALID | FE_OVERFLOW);
}

// ########################################################
// ########################################################


// ########################################################
//                       Destructor
// ########################################################
//

PairPANNA::~PairPANNA()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

// ########################################################
// ########################################################

// Radial gvect contribution (and derivative part)
double PairPANNA::Gradial_d(double rdiff, int indr, double *dtmp){
  double cent = rdiff - par.Rsi_rad[indr];
  double gauss = exp( - par.eta_rad[indr] * cent * cent);
  double fc = 0.5 * ( 1.0 + cos(rdiff * par.iRc_rad) );
  *dtmp = ( par.iRc_rad_half * sin(rdiff * par.iRc_rad) +
         par.twoeta_rad[indr] * fc * cent ) * gauss / rdiff;
  return gauss * fc;
}

// Angular gvect contribution (and derivative part)
double PairPANNA::Gangular_d(double rdiff1, double rdiff2, double cosijk, int Rsi, int Thi, double* dtmp){
  if(cosijk> 0.999999999) cosijk =  0.999999999;
  if(cosijk<-0.999999999) cosijk = -0.999999999;
  double epscorr = 0.001;
  double sinijk = sqrt(1.0 - cosijk*cosijk + epscorr * pow(par.Thi_sin[Thi], 2) );
  double iRij = 1.0/rdiff1;
  double iRik = 1.0/rdiff2;
  double Rcent = 0.5 * (rdiff1 + rdiff2) - par.Rsi_ang[Rsi];
  double fcrad = 0.5 * ( 1.0 + par.Thi_cos[Thi] * cosijk + par.Thi_sin[Thi] * sinijk );
  double fcij = 0.5 * ( 1.0 + cos(rdiff1 * par.iRc_ang) );
  double fcik = 0.5 * ( 1.0 + cos(rdiff2 * par.iRc_ang) );
  double mod_norm = pow( 0.5 * (1.0 + sqrt(1.0 + epscorr * pow(par.Thi_sin[Thi], 2) ) ), par.zeta[Thi]);
  double fact0 = 2.0 * exp( - par.eta_ang[Rsi] * Rcent * Rcent) * pow(fcrad, par.zeta[Thi]-1) / mod_norm;
  double fact1 = fact0 * fcij * fcik;
  double fact2 = par.zeta_half[Thi] * fact1 * ( par.Thi_cos[Thi] - par.Thi_sin[Thi] * cosijk / sinijk );
  double fact3 = par.iRc_ang_half * fact0 * fcrad;
  double G = fact1 * fcrad;
  dtmp[0] = -iRij * ( par.eta_ang[Rsi] * Rcent * G
            + fact2 * cosijk * iRij
            + fact3 * fcik * sin(rdiff1 * par.iRc_ang) );
  dtmp[1] = fact2 * iRij * iRik;
  dtmp[2] = -iRik * ( par.eta_ang[Rsi] * Rcent * G
            + fact2 * cosijk * iRik
            + fact3 * fcij * sin(rdiff2 * par.iRc_ang) );
  return G;
}

// Function computing gvect and its derivative
void PairPANNA::compute_gvect(int ind1, double **x, int* type,
                              int* neighs, int num_neigh,
                              double *G, double* dGdx){
  float posx = x[ind1][0];
  float posy = x[ind1][1];
  float posz = x[ind1][2];
  // Elements to store neigh list for angular part
  // We allocate max possible size, so we don't need to reallocate
  int nan = 0;
  int ang_neigh[num_neigh];
  int ang_type[num_neigh];
  double dists[num_neigh];
  double diffx[num_neigh];
  double diffy[num_neigh];
  double diffz[num_neigh];
  //
  // Loop on neighbours, compute radial part, store quantities for angular
  for(int n=0; n<num_neigh; n++){
    int nind = neighs[n];
    double dx = x[nind][0]-posx;
    double dy = x[nind][1]-posy;
    double dz = x[nind][2]-posz;
    double Rij = sqrt(dx*dx+dy*dy+dz*dz);
    if (Rij < par.Rc_rad){
      // Add all radial parts
      int indsh = (type[nind]-1)*par.RsN_rad;
      for(int indr=0; indr<par.RsN_rad; indr++){
        double dtmp;
        // Getting the simple G and derivative part
        G[indsh+indr] += Gradial_d(Rij, indr, &dtmp);
        // Filling all derivatives
        int indsh2 = (indsh+indr)*(num_neigh+1)*3;
        double derx = dtmp*dx;
        double dery = dtmp*dy;
        double derz = dtmp*dz;
        dGdx[indsh2 + num_neigh*3     ] += derx;
        dGdx[indsh2 + num_neigh*3 + 1 ] += dery;
        dGdx[indsh2 + num_neigh*3 + 2 ] += derz;
        dGdx[indsh2 + n*3     ] -= derx;
        dGdx[indsh2 + n*3 + 1 ] -= dery;
        dGdx[indsh2 + n*3 + 2 ] -= derz;
      }
    }
    // If within radial cutoff, store quantities
    if (Rij < par.Rc_ang){
      ang_neigh[nan] = n;
      ang_type[nan] = type[nind];
      dists[nan] = Rij;
      diffx[nan] = dx;
      diffy[nan] = dy;
      diffz[nan] = dz;
      nan++;
    }
  }

  // Loop on angular neighbours and fill angular part
  for(int n=0; n<nan-1; n++){
    for(int m=n+1; m<nan; m++){
      // Compute cosine
      double cos_ijk = (diffx[n]*diffx[m] + diffy[n]*diffy[m] + diffz[n]*diffz[m]) /
                       (dists[n]*dists[m]);
      // Gvect shift due to species
      int indsh = par.typsh[ang_type[n]-1][ang_type[m]-1];
      // Loop over all bins
      for(int Rsi=0; Rsi<par.RsN_ang; Rsi++){
        for(int Thi=0; Thi<par.ThetasN; Thi++){
          double dtmp[3];
          int indsh2 = Rsi * par.ThetasN + Thi;
          // Adding the G part and computing derivative
          G[indsh+indsh2] += Gangular_d(dists[n], dists[m], cos_ijk, Rsi, Thi, dtmp);
          // Computing the derivative contributions
          double dgdxj = dtmp[0]*diffx[n] + dtmp[1]*diffx[m];
          double dgdyj = dtmp[0]*diffy[n] + dtmp[1]*diffy[m];
          double dgdzj = dtmp[0]*diffz[n] + dtmp[1]*diffz[m];
          double dgdxk = dtmp[1]*diffx[n] + dtmp[2]*diffx[m];
          double dgdyk = dtmp[1]*diffy[n] + dtmp[2]*diffy[m];
          double dgdzk = dtmp[1]*diffz[n] + dtmp[2]*diffz[m];
          // Filling all the interested terms
          int indsh3 = (indsh+indsh2)*(num_neigh+1)*3;
          dGdx[indsh3 + ang_neigh[n]*3     ] += dgdxj;
          dGdx[indsh3 + ang_neigh[n]*3 + 1 ] += dgdyj;
          dGdx[indsh3 + ang_neigh[n]*3 + 2 ] += dgdzj;
          dGdx[indsh3 + ang_neigh[m]*3     ] += dgdxk;
          dGdx[indsh3 + ang_neigh[m]*3 + 1 ] += dgdyk;
          dGdx[indsh3 + ang_neigh[m]*3 + 2 ] += dgdzk;
          dGdx[indsh3 + num_neigh*3     ] -= dgdxj + dgdxk;
          dGdx[indsh3 + num_neigh*3 + 1 ] -= dgdyj + dgdyk;
          dGdx[indsh3 + num_neigh*3 + 2 ] -= dgdzj + dgdzk;
        }
      }
    }
  }

}

double PairPANNA::compute_network(double *G, double *dEdG, int type){
// double PairPANNA::compute_network(double *G, double *dEdG, int type, int inum){
  // *1 layer input
  // *2 layer output
  double *lay1, *lay2, *dlay1, *dlay2;
  dlay1 = new double[par.layers_size[type][0]*par.gsize];

  // std::ofstream myfile;
  // std::cout << "computation: " << inum << std::endl;
  // myfile.open("atom_" + std::to_string(inum) +".dat");
  lay1 = G;
  // myfile << "=G=,";
  // for(int i = 0; i <par.gsize; i++){
  //   myfile << lay1[i]<< ',';
  // }
  // myfile << std::endl;

  for(int i=0; i<par.layers_size[type][0]*par.gsize; i++) dlay1[i] = 0.0;
  // dG_i/dG_i = 1
  for(int i=0; i<par.gsize; i++) dlay1[i*par.gsize+i] = 1.0;
  // Loop over layers
  for(int l=0; l<par.Nlayers[type]; l++){
    int size1 = par.layers_size[type][l];
    int size2 = par.layers_size[type][l+1];
    lay2 = new double[size2];
    dlay2 = new double[size2*par.gsize];
    for(int i=0; i<size2*par.gsize; i++) dlay2[i]=0.0;
    // Matrix vector multiplication done by hand for now...
    // We compute W.x+b and W.(dx/dg)
    for(int i=0; i<size2; i++){
      // a_i = b_i
      lay2[i] = network[type][2*l+1][i];
      for(int j=0;j<size1; j++){
        // a_i += w_ij * x_j
        lay2[i] += network[type][2*l][i*size1+j]*lay1[j];
        // lay2[i] += network[type][2*l][j*size2+i]*lay1[j];
        for(int k=0; k<par.gsize; k++)
          // da_i/dg_k += w_ij * dx_j/dg_k
          dlay2[i*par.gsize+k] += network[type][2*l][i*size1+j]*dlay1[j*par.gsize+k];
          // dlay2[i*par.gsize+k] += network[type][2*l][j*size2+i]*dlay1[j*par.gsize+k];
      }
    }

    // here dump of W.x+b
    // myfile << "=L" << std::to_string(l) << "=,";
    // for(int i = 0; i <size2; i++){
    //   myfile << lay2[i]<< ',';
    // }
    // myfile << std::endl;

    // Apply appropriate activation
    // Gaussian
    if(par.layers_activation[type][l]==1){
      for(int i=0; i<size2; i++){
        double tmp = exp(-lay2[i]*lay2[i]);
        for(int k=0; k<par.gsize; k++)
          dlay2[i*par.gsize+k] *= -2.0*lay2[i]*tmp;
        lay2[i] = tmp;
      }
    }
    // ReLU
    else if(par.layers_activation[type][l]==3){
      for(int i=0; i<size2; i++){
        if(lay2[i]<0){
          lay2[i] = 0.0;
          for(int k=0; k<par.gsize; k++) dlay2[i*par.gsize+k] = 0.0;
        }
      }
    }
    // Tanh
    else if(par.layers_activation[type][l]==4){
      for(int i=0; i<size2; i++){
        double tmp = tanh(lay2[i]);
        for(int k=0; k<par.gsize; k++)
          dlay2[i*par.gsize+k] *= (1 - tmp * tmp);
        lay2[i] = tmp;
      }
    }
    // Otherwise it's linear and nothing needs to be done

    if(l!=0) delete[] lay1;
    delete[] dlay1;
    lay1 = lay2;
    dlay1 = dlay2;
  }
  // myfile.close();
  for(int i=0;i<par.gsize;i++) dEdG[i]=dlay1[i];
  double E = lay1[0];
  delete[] lay1;
  delete[] dlay1;
  return E;
}

// ########################################################
//                       COMPUTE
// ########################################################
// Determine the energy and forces for the current structure.

void PairPANNA::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  // I'll assume the order is the same.. we'll need to create a mapping if not the case
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // Looping on local atoms
  for(int a=0; a<inum; a++){
    int myind = ilist[a];
    // Allocate this gvect and dG/dx and zero them
    double G[par.gsize];
    double dEdG[par.gsize];
    // dGdx has (numn+1)*3 derivs per elem: neigh first, then the atom itself
    double dGdx[par.gsize*(numneigh[myind]+1)*3];
    for(int i=0; i<par.gsize; i++){
      G[i] = 0.0;
      for(int j=0; j<(numneigh[myind]+1)*3; j++)
        dGdx[i*(numneigh[myind]+1)*3+j] = 0.0;
    }
    // Calculate Gvect and derivatives
    compute_gvect(myind, x, type, firstneigh[myind], numneigh[myind], G, dGdx);

    // Apply network
    double E = compute_network(G,dEdG,type[myind]-1);
    // Calculate forces
    int shift = (numneigh[myind]+1)*3;
    for(int n=0; n<numneigh[myind]; n++){
      int nind = firstneigh[myind][n];
      for(int j=0; j<par.gsize; j++){
        f[nind][0] -= dEdG[j]*dGdx[j*shift + 3*n    ];
        f[nind][1] -= dEdG[j]*dGdx[j*shift + 3*n + 1];
        f[nind][2] -= dEdG[j]*dGdx[j*shift + 3*n + 2];
      }
    }
    for(int j=0; j<par.gsize; j++){
      f[myind][0] -= dEdG[j]*dGdx[j*shift + 3*numneigh[myind]    ];
      f[myind][1] -= dEdG[j]*dGdx[j*shift + 3*numneigh[myind] + 1];
      f[myind][2] -= dEdG[j]*dGdx[j*shift + 3*numneigh[myind] + 2];
    }

    if (eflag_global) eng_vdwl += E;
    if (eflag_atom) eatom[myind] += E;
  }

  if (vflag_fdotr) {
    virial_fdotr_compute();
  }

}

// ########################################################
// ########################################################

// Get a new line skipping comments or empty lines
// Set value=... if [...], return 1
// Fill key,value if 'key=value', return 2
// Set value=... if ..., return 3
// Return 0 if eof, <0 if error, >0 if okay
int PairPANNA::get_input_line(std::ifstream* file, std::string* key, std::string* value){
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

int PairPANNA::get_parameters(char* directory, char* filename)
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
  par.Nspecies = -1;
  // Flags to keep track of set parameters
  int Npars = 17;
  int parset[Npars];
  for(int i=0;i<Npars;i++) parset[i]=0;
  int *spset;
  std::string version = "v0"; int gversion=0;
  double tmp_eta_rad; double tmp_eta_ang ; int tmp_zeta ;
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
      // Gvect param section
      if(value=="GVECT_PARAMETERS"){ section = 0; }
      // For now other sections are just species networks
      else {
        // First time after params are read: do checks
        if(section==0){
          if (gversion==0){
            if(parset[5]==0){
            // Set steps if they were omitted
            par.Rsst_rad = (par.Rc_rad - par.Rs0_rad) / par.RsN_rad; parset[5]=1;}
            if(parset[10]==0){
            par.Rsst_ang = (par.Rc_ang - par.Rs0_ang) / par.RsN_ang; parset[10]=1;}}
          else if (gversion==1){parset[5]=1 ; parset[10]=1;}
          // Check that all parameters have been set
          for(int p=0;p<Npars;p++){
            if(parset[p]==0){
              std::cout << "Parameter " << p << " not set!" << std::endl;  return -1; } }
          // Calculate Gsize
          par.gsize = par.Nspecies * par.RsN_rad + (par.Nspecies*(par.Nspecies+1))/2 * par.RsN_ang * par.ThetasN;
        } //section 0 ended
        int match = 0;
        for(int s=0;s<par.Nspecies;s++){
          // If species matches the list, change section
          if(value==par.species[s]){
            section = s+1;
            match = 1;
          }
        }
        if(match==0){
          std::cout << "Species " << value << " not found in species list." << std::endl;
          return -2;
        }
      }
    }// A header is parsed
    else if(parseint==2){
      // Parse param section
      if(section==0){
	std::string comma = ",";
        if(key=="Nspecies"){
          par.Nspecies = std::atoi(value.c_str());
          // Small check
          if(par.Nspecies<1){
            std::cout << "Nspecies needs to be >0." << std::endl;
            return -2; }
          parset[0] = 1;
          // Allocate species list
          par.species = new std::string[par.Nspecies];
          // Allocate network quantities
          par.Nlayers = new int[par.Nspecies];
          par.layers_size = new int*[par.Nspecies];
          par.layers_activation = new int*[par.Nspecies];
          network = new double**[par.Nspecies];
          // Keep track of set species
          spset = new int[par.Nspecies];
          for(int s=0;s<par.Nspecies;s++) {
            par.Nlayers[s] = -1;
            spset[s]=0; } }
        else if(key=="species"){
          //std::string comma = ",";
          size_t pos = 0;
          int s = 0;
          // Parse species list
          while ((pos = value.find(comma)) != std::string::npos) {
            if(s>par.Nspecies-2){
              std::cout << "Species list longer than Nspecies." << std::endl;
              return -2; }
            par.species[s] = value.substr(0, pos);
            value.erase(0, pos+1);  s++; }
          if(value.length()>0){
            par.species[s] = value; s++; };
          if(s<par.Nspecies){
            std::cout << "Species list shorter than Nspecies." << std::endl;
            return -2; }
          parset[1] = 1; }
        // Common features are read.
        // From here on what will be read depends on the gversion
        if(gversion == 0){ // Potentials compatible with OPENKIM
          std::cout << "G Version is " << gversion << std::endl;
          if(key=="eta_rad"){
            tmp_eta_rad = std::atof(value.c_str()); parset[2] = 0; }
          else if(key=="Rc_rad"){
            par.Rc_rad = std::atof(value.c_str()); parset[3] = 1; }
          else if(key=="Rs0_rad"){
            par.Rs0_rad = std::atof(value.c_str());  parset[4] = 1; }
          else if(key=="Rsst_rad"){
            par.Rsst_rad = std::atof(value.c_str()); parset[5] = 1; }
          else if(key=="RsN_rad"){
            par.RsN_rad = std::atoi(value.c_str()); parset[6] = 1; 
            par.eta_rad = new float[par.RsN_rad]; 
            par.twoeta_rad = new float[par.RsN_rad];
            par.Rs_rad  = new float[par.RsN_rad];
            for(int i=0;i<par.RsN_rad;i++) par.eta_rad[i]=tmp_eta_rad;
            for(int i=0;i<par.RsN_rad;i++) par.Rs_rad[i]= par.Rs0_rad + i *(par.Rc_rad - par.Rs0_rad) / par.RsN_rad ; 
            parset[14]=1; parset[2]=1;}
          else if(key=="eta_ang"){
            tmp_eta_ang = std::atof(value.c_str()); parset[7] = 0; }
          else if(key=="Rc_ang"){
            par.Rc_ang = std::atof(value.c_str()); parset[8] = 1; }
          else if(key=="Rs0_ang"){
            par.Rs0_ang = std::atof(value.c_str()); parset[9] = 1; }
          else if(key=="Rsst_ang"){
            par.Rsst_ang = std::atof(value.c_str()); parset[10] = 1; }
          else if(key=="RsN_ang"){
            par.RsN_ang = std::atoi(value.c_str()); parset[11] = 1; 
            par.eta_ang = new float[par.RsN_ang];
            par.Rs_ang  = new float[par.RsN_ang];
            for(int i=0;i<par.RsN_ang;i++) par.eta_ang[i]=tmp_eta_ang;
            for(int i=0;i<par.RsN_ang;i++) par.Rs_ang[i]= par.Rs0_ang + i *(par.Rc_ang - par.Rs0_ang) / par.RsN_ang ; 
            parset[15]=1; parset[7]=1;}
          else if(key=="zeta"){
            tmp_zeta = std::atof(value.c_str()); parset[12] = 0; }
          else if(key=="ThetasN"){
            par.ThetasN = std::atoi(value.c_str()); parset[13] = 1; 
            par.zeta = new int[par.ThetasN];
            par.zeta_half = new float[par.ThetasN];
            par.Thetas = new float[par.ThetasN];
            for(int i=0;i<par.ThetasN;i++) par.zeta[i]=tmp_zeta; parset[12]=1;
            for(int i=0;i<par.ThetasN;i++) par.Thetas[i]= (0.5f+ i)*(M_PI/par.ThetasN); parset[16]=1;}  
        }//gversion = 0
        else if(gversion ==1 ){
          //First read allocation sizes
          if(key=="RsN_rad"){
            par.RsN_rad = std::atoi(value.c_str());
            par.eta_rad = new float[par.RsN_rad];
	          par.twoeta_rad = new float[par.RsN_rad];
            par.Rs_rad = new float[par.RsN_rad]; parset[6]=1;}
          // Then param arrays
          else if(key=="eta_rad"){
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.eta_rad[s] = std::atof(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.eta_rad[s] = std::atof(value.c_str()); s++; }; parset[2] = 1; }
          // Then cutoff
          else if(key=="Rc_rad"){
            par.Rc_rad = std::atof(value.c_str()); parset[3] = 1; }
          // Then the bin center arrays
          else if(key=="Rs_rad"){
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.Rs_rad[s] = std::atof(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.Rs_rad[s] = std::atof(value.c_str()); s++; }; parset[14] = 1;
            par.Rs0_rad=par.Rs_rad[0];                           parset[4]=1;}

          else if(key=="RsN_ang"){
            par.RsN_ang = std::atoi(value.c_str());
            par.eta_ang = new float[par.RsN_ang];
            par.Rs_ang = new float[par.RsN_ang]; parset[11]=1;}
          else if(key=="eta_ang") {
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.eta_ang[s] = std::atof(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.eta_ang[s] = std::atof(value.c_str()); s++; }; parset[7] = 1; }

          // Then cutoffs
          else if(key=="Rc_ang"){
            par.Rc_ang = std::atof(value.c_str()); parset[8] = 1; }
          // Then the bin center arrays
          else if(key=="Rs_ang") {
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.Rs_ang[s] = std::atof(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.Rs_ang[s] = std::atof(value.c_str()); s++; }; parset[15] = 1;
            par.Rs0_ang=par.Rs_ang[0];                           parset[9]=1;}

          else if(key=="ThetasN"){
            par.ThetasN = std::atoi(value.c_str());
            par.zeta = new int[par.ThetasN];
            par.zeta_half = new float[par.ThetasN];
            par.Thetas = new float[par.ThetasN]; parset[13]=1;}
          else if(key=="zeta") {
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.zeta[s] = std::atoi(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.zeta[s] = std::atoi(value.c_str()); s++; }; parset[12] = 1; }

          else if(key=="Thetas") {
            size_t pos = 0; int s = 0;
            value=value.substr(1, value.size() - 2); //get rid of [ ]
            while ((pos = value.find(comma)) != std::string::npos) {
              par.Thetas[s] = std::atof(value.substr(0, pos).c_str());
              value.erase(0, pos+1);  s++; }
            if(value.length()>0){ par.Thetas[s] = std::atof(value.c_str()); s++; };  parset[16] = 1; }
        } //gversion = 1
      } //Section 0 (Parameter parsing) is finished.
      // Parse species network
      else if(section<par.Nspecies+1){
        int s=section-1;
        // Read species network
        if(key=="Nlayers"){
          par.Nlayers[s] = std::atoi(value.c_str());
          // This has the extra gvect size
          par.layers_size[s] = new int[par.Nlayers[s]+1];
          par.layers_size[s][0] = par.gsize;
          par.layers_size[s][1] = 0;
          par.layers_activation[s] = new int[par.Nlayers[s]];
          for(int i=0;i<par.Nlayers[s]-1;i++) par.layers_activation[s][i]=1;
          par.layers_activation[s][par.Nlayers[s]-1]=0;
          network[s] = new double*[2*par.Nlayers[s]];
        }
        else if(key=="sizes"){
          if(par.Nlayers[s]==-1){
            std::cout << "Sizes cannot be set before Nlayers." << std::endl;
            return -3;
          }
          std::string comma = ",";
          size_t pos = 0;
          int l = 0;
          // Parse layers list
          while ((pos = value.find(comma)) != std::string::npos) {
            if(l>par.Nlayers[s]-2){
              std::cout << "Layers list longer than Nlayers." << std::endl;
              return -3;
            }
            std::string lsize = value.substr(0, pos);
            par.layers_size[s][l+1] = std::atoi(lsize.c_str());
            value.erase(0, pos+1);
            l++;
          }
          if(value.length()>0){
            par.layers_size[s][l+1] = std::atoi(value.c_str());
            l++;
          };
          if(l<par.Nlayers[s]){
            std::cout << "Layers list shorter than Nlayers." << std::endl;
            return -3;
          }
        }
        else if(key=="activations"){
          if(par.Nlayers[s]==-1){
            std::cout << "Activations cannot be set before Nlayers." << std::endl;
            return -3;
          }
          std::string comma = ",";
          size_t pos = 0;
          int l = 0;
          // Parse layers list
          while ((pos = value.find(comma)) != std::string::npos) {
            if(l>par.Nlayers[s]-2){
              std::cout << "Activations list longer than Nlayers." << std::endl;
              return -3;
            }
            std::string lact = value.substr(0, pos);
            int actnum = std::atoi(lact.c_str());
            if (actnum!=0 && actnum!=1 && actnum!=3 && actnum!=4 ){
              std::cout << "Activations unsupported: " << actnum << std::endl;
              return -3;
            }
            par.layers_activation[s][l] = actnum;
            value.erase(0, pos+1);
            l++;
          }
          if(value.length()>0){
            int actnum = std::atoi(value.c_str());
            if (actnum!=0 && actnum!=1 && actnum!=3 && actnum!=4){
              std::cout << "Activations unsupported: " << actnum << std::endl;
              return -3;
            }
            par.layers_activation[s][l] = actnum;
            l++;
          };
          if(l<par.Nlayers[s]){
            std::cout << "Activations list shorter than Nlayers." << std::endl;
            return -3;
          }
        }
        else if(key=="file"){
          if(par.layers_size[s][1]==0){
            std::cout << "Layers sizes unset before filename for species " << par.species[s] << std::endl;
            return -3;
          }
          // Read filename and load weights
          wfile_string = dir_string+"/"+value;
          weights_file.open(wfile_string.c_str(), std::ios::binary);
          if(!weights_file.is_open()){
            std::cout << "Error reading weights file for " << par.species[s] << std::endl;
            return -3;
          }
          for(int l=0; l<par.Nlayers[s]; l++){
            // Allocate and read the right amount of data
            // Weights
            network[s][2*l] = new double[par.layers_size[s][l]*par.layers_size[s][l+1]];
            for(int i=0; i<par.layers_size[s][l]; i++) {
              for(int j=0; j<par.layers_size[s][l+1]; j++) {
                float num;
                weights_file.read(reinterpret_cast<char*>(&num), sizeof(float));
                if(weights_file.eof()){
                  std::cout << "Weights file " << wfile_string << " is too small." << std::endl;
                  return -3;
                }
                network[s][2*l][j*par.layers_size[s][l]+i] = (double)num;
              }
            }
            // Biases
            network[s][2*l+1] = new double[par.layers_size[s][l+1]];
            for(int d=0; d<par.layers_size[s][l+1]; d++) {
              float num;
              weights_file.read(reinterpret_cast<char*>(&num), sizeof(float));
              if(weights_file.eof()){
                std::cout << "Weights file " << wfile_string << " is too small." << std::endl;
                return -3;
              }
              network[s][2*l+1][d] = (double)num;
            }
          }
          // Check if we're not at the end
          std::ifstream::pos_type fpos = weights_file.tellg();
          weights_file.seekg(0, std::ios::end);
          std::ifstream::pos_type epos = weights_file.tellg();
          if(fpos!=epos){
            std::cout << "Weights file " << wfile_string << " is too big." << std::endl;
            return -3;
          }
          weights_file.close();
          spset[section-1] = 1;
        }
      }
      else{
        return -3;
      }
    }
    else if(parseint==3){
      // Version information is read:
      if(key == "!version") {
        version = value ;
        std::cout << "Network version " << value << std::endl; }
      else if(key == "!gversion") {
        gversion = std::atoi(value.c_str()) ;
        std::cout << "Gvector version " << value << std::endl;}
    }
    // Get new line
    parseint = get_input_line(&params_file,&key,&value);
  }

  // Derived params - for both gvect types done here
  par.cutmax = par.Rc_rad>par.Rc_ang ? par.Rc_rad : par.Rc_ang;
  for(int i=0; i<par.RsN_rad; i++) {
    par.twoeta_rad[i] = 2.0*par.eta_rad[i];}
  for(int i=0; i<par.ThetasN; i++) {
    par.zeta_half[i] = 0.5f*par.zeta[i];}
  par.iRc_rad = M_PI/par.Rc_rad;
  par.iRc_rad_half = 0.5*par.iRc_rad;
  par.iRc_ang = M_PI/par.Rc_ang;
  par.iRc_ang_half = 0.5*par.iRc_ang;
  //par.Rsi_rad = new float[par.RsN_rad];
  //for(int indr=0; indr<par.RsN_rad; indr++) par.Rsi_rad[indr] = par.Rs0_rad + indr * par.Rsst_rad;
  par.Rsi_rad = par.Rs_rad;
  //par.Rsi_ang = new float[par.RsN_ang];
  //for(int indr=0; indr<par.RsN_ang; indr++) par.Rsi_ang[indr] = par.Rs0_ang + indr * par.Rsst_ang;
  par.Rsi_ang = par.Rs_ang;
  par.Thi_cos = new float[par.ThetasN];
  par.Thi_sin = new float[par.ThetasN];
  for(int indr=0; indr<par.ThetasN; indr++)  {
    //float ti = (indr + 0.5f) * M_PI / par.ThetasN;
    float ti = par.Thetas[indr];
    par.Thi_cos[indr] = cos(ti);
    par.Thi_sin[indr] = sin(ti);
  }
  for(int s=0;s<par.Nspecies;s++){
    if(spset[s]!=1){
      std::cout << "Species network undefined for " << par.species[s] << std::endl;
      return -4;
    }
  }

  // Precalculate gvect shifts for any species pair
  par.typsh = new int*[par.Nspecies];
  for(int s=0; s<par.Nspecies; s++){
    par.typsh[s] = new int[par.Nspecies];
    for(int ss=0; ss<par.Nspecies; ss++){
      if(s<ss) par.typsh[s][ss] = par.Nspecies*par.RsN_rad +
                  (s*par.Nspecies - (s*(s+1))/2 + ss) *
                  par.RsN_ang * par.ThetasN;
      else par.typsh[s][ss] = par.Nspecies*par.RsN_rad +
                  (ss*par.Nspecies - (ss*(ss+1))/2 + s) *
                  par.RsN_ang * par.ThetasN;
    }
  }
  params_file.close();
  delete[] spset;
  return(0);
}

// ########################################################
//                       ALLOCATE
// ########################################################
// Allocates all necessary arrays.

void PairPANNA::allocate()
{

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 1;
    }
  }
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
}

// ########################################################
// ########################################################

// ########################################################
//                       COEFF
// ########################################################
// Load all the gvectors and NN parameters

void PairPANNA::coeff(int narg, char **arg)
{

  if (!allocated) {
    allocate();
  }

  // We now expect a directory and the parameters file name (inside the directory) with all params
  if (narg != 4) {
    error->all(FLERR,"Format of pair_coeff command is\npair_coeff * *  network_directory parameter_file\n");
  }

  std::cout << "Loading PANNA pair parameters from " << arg[2] << "/" << arg[3] << std::endl;
  int gpout = get_parameters(arg[2], arg[3]);
  if(gpout==0){
    std::cout << "Network loaded!" << std::endl;
  }
  else{
    std::cout << "Error " << gpout << " while loading network!" << std::endl;
    exit(1);
  }

  for (int i=1; i<=atom->ntypes; i++) {
    for (int j=1; j<=atom->ntypes; j++) {
      cutsq[i][j] = par.cutmax * par.cutmax;
    }
  }
}

// ########################################################
// ########################################################

// ########################################################
//                       INIT_STYLE
// ########################################################
// Set up the pair style to be a NN potential.

void PairPANNA::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style PANNA requires newton pair on");

  int irequest;
  neighbor->cutneighmin = 1.0;
  neighbor->cutneighmax = par.cutmax;
  neighbor->delay = 0;
  neighbor->every = 10;
  neighbor->skin = 1.0;
  irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->id=1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 0;

}

// ########################################################
// ########################################################

// ########################################################
//                       INIT_LIST
// ########################################################
//

void PairPANNA::init_list(int id, NeighList *ptr)
{
  if(id == 1) {
    list = ptr;
  }
}

// ########################################################
// ########################################################

// ########################################################
//                       init_one
// ########################################################
// Initilize 1 pair interaction.  Needed by LAMMPS but not
// used in this style.

double PairPANNA::init_one(int i, int j)
{
  return sqrt(cutsq[i][j]);
}

// ########################################################
// ########################################################



// ########################################################
//                       WRITE_RESTART
// ########################################################
// Writes restart file. Not implemented.

void PairPANNA::write_restart(FILE *fp)
{

}

// ########################################################


// ########################################################
//                       READ_RESTART
// ########################################################
// Reads from restart file. Not implemented.

void PairPANNA::read_restart(FILE *fp)
{

}

// ########################################################


// ########################################################
//                       WRITE_RESTART_SETTINGS
// ########################################################
// Writes settings to restart file. Not implemented.

void PairPANNA::write_restart_settings(FILE *fp)
{

}

// ########################################################
// ########################################################



// ########################################################
//                       READ_RESTART_SETTINGS
// ########################################################
// Reads settings from restart file. Not implemented.

void PairPANNA::read_restart_settings(FILE *fp)
{

}

// ########################################################
// ########################################################

// Not implemented.
void PairPANNA::write_data(FILE *fp)
{
  /*
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
  */
}

// Not implemented.
void PairPANNA::write_data_all(FILE *fp)
{
  /*
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
  */
}

// Not implemented.
double PairPANNA::single(int i, int j, int itype, int jtype, double rsq,
                      double factor_coul, double factor_lj,
                      double &fforce)
{
  return 1;
}

/* ---------------------------------------------------------------------- */



// ########################################################
//                       Settings
// ########################################################
// Initializes settings. No setting needed.

void PairPANNA::settings(int narg, char* argv[])
{
  if (narg != 0) {
    error->all(FLERR,"pair_panna requires no arguments.\n");
  }

}

// ########################################################
// ########################################################
