# PANNA
## Properties from Artificial Neural Network Architectures

> See the included work-in-progress **tutorial** to have a walkthrough of PANNA, 
> the tutorial can be run from command line, or through jupyter notebooks.
> 
> A version of the documentation and tutorial is also available at `https://pannadevs.gitlab.io/pannadoc`.


PANNA is a package for training and validating neural networks to represent atomic potentials. 
It implements configurable all-to-all connected deep neural network architectures 
which allow for the exploration of training dynamics.
Currently it includes tools to enable original[1] and modified[2] Behler-Parrinello input feature vectors,
both for molecules and crystals,
but the network can also be used in an input-agnostic fashion to enable further experimentation.
PANNA is written in Python and relies on TensorFlow as underlying engine.

A common way to use PANNA in its current implementation is 
to train a neural network in order to estimate the total energy of a molecule or crystal, 
as a sum of atomic contributions,
by learning from the data of reference total energy calculations for similar structures (usually *ab-initio* calculations).

The neural network models in literature often start from a description 
of the system of interest in terms of local *feature vectors* for each atom in the configuration.
PANNA provides tools to calculate two versions of the Behler-Parrinello local descriptors but
it allows the use of any species-resolved, fixed-size array that describes the input data.

PANNA allows the construction of neural network architectures with different sizes for each of the atomic species in the training set. 
Currently the allowed architecture is a deep neural network of fully connected layers, 
starting from the input feature vector and going through one or more *hidden layers*. 
The user can determine to train or freeze any layer, s/he can also transfer network parameters between species upon restart.

In the present implementation, the network layers are connected through a linear operation plus a nonlinear activation function. 
As an example, with the default Gaussian activation, node `j` of layer `i` will depend on the values of the nodes `k` of layer `i-1` as:
```
l[i,j] = Exp[ -(Sum_k (w[i,j,k]*l[i-1,k]) + b[i,j])^2 ],
```
where the *weights* `w[i,j,k]` and the *biases* `b[i,j]` are the trainable parameters of the model.
The energy contribution of each atom is then obtained as a final layer of a single node with linear activation, i.e.:
```
E = Sum_k (w[N,k]*l[N-1,k]) + b[N].
```
The estimated energy is the sum of all atomic contributions in the system.

In summary, PANNA is an easy-to-use interface for obtaining neural network models for atomistic potentials, 
 leveraging the highly optimized TensorFlow infrastructure to provide an efficient and parallelized, 
GPU-accelerated training. 

The complete toolchain allows the user to utilize Quantum Espresso data files for training and
deploy the final neural network as a LAMMPS potential for further simulations.

If you have any questions you can write to
  * panna google groups
https://groups.google.com/d/forum/pannausers
  * pannadevelopers@gmail.com
  * kucukben@gmail.com (if you use this last option please add [pannadevs]
with the square brackets to your subject line)


REFERENCES

    [1] J. Behler and M. Parrinello, Generalized Neural-Network
    Representation  of  High-Dimensional  Potential-Energy
    Surfaces, Phys. Rev. Lett. 98, 146401 (2007)
    [2] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg.
    ANI-1: An extensible neural network potential with DFT accuracy
    at force field computational cost. Chemical Science,(2017), DOI: 10.1039/C6SC05720A
