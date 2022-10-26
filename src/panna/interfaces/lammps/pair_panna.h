//###########################################################################
//# Copyright (c), The PANNAdevs group. All rights reserved.                #
//# This file is part of the PANNA code.                                    #
//#                                                                         #
//# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
//# For further information on the license, see the LICENSE.txt file        #
//###########################################################################
#ifdef PAIR_CLASS

PairStyle(panna, PairPANNA)

#else

#ifndef LMP_PAIR_PANNA_H
#define LMP_PAIR_PANNA_H

#include "pair.h"
#include <string>
#include <fstream>

namespace LAMMPS_NS {

class PairPANNA : public Pair {
public:
  PairPANNA(class LAMMPS *);
  virtual ~PairPANNA();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);

  int get_parameters(char*, char *);
  int get_input_line(std::ifstream*, std::string*, std::string*);

  void compute_gvect(int, double**, int*, int*, int, double*, double*);
  double Gradial_d(double, int, double*);
  double Gangular_d(double, double, double, int, int, double*);
  double compute_network(double*, double*, int);
  // double compute_network(double*, double*, int, int);

  struct parameters{
    int Nspecies;
    // Gvector parameters
    float *eta_rad;
    float Rc_rad;
    float Rs0_rad;
    float Rsst_rad;
    int RsN_rad;
    float *eta_ang;
    float Rc_ang;
    float Rs0_ang;
    float Rsst_ang;
    int RsN_ang;
    int *zeta;
    int ThetasN;
    std::string* species;
    int gsize;
    float * Rs_rad;
    float * Rs_ang;
    float * Thetas;

    // Network parameters
    int *Nlayers;
    int **layers_size;
    int **layers_activation;

    // Useful precalculated quantities
    float cutmax;
    float *twoeta_rad;
    float *zeta_half;
    float iRc_rad;
    float iRc_rad_half;
    float iRc_ang;
    float iRc_ang_half;
    float *Rsi_rad;
    float *Rsi_ang;
    float *Thi_cos;
    float *Thi_sin;
    int **typsh;
  };


protected:
  // Gvect and NN parameters
  struct parameters par;
  // The network [species, layers, array]
  double ***network;

  void allocate();

};

}

#endif
#endif
