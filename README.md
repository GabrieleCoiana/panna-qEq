## PANNA
### Properties from Artificial Neural Network Architectures

### For the general structure (PANNA 1.0) see publication at: [Comp Phys Comm, 256, 107402 (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010465520301843)
### or the pre-review version at: [arXiv:1907.03055](https://arxiv.org/abs/1907.03055)
### For the latest version (PANNA 2.0) see publication in: [J. Chem. Phys. 159, 084117 (2023)](https://pubs.aip.org/aip/jcp/article/159/8/084117/2908459/PANNA-2-0-Efficient-neural-network-interatomic)
### or the pre-review version at: [arXiv:2305.11805](https://arxiv.org/abs/2305.11805)
### For PANNA-JAX and the LATTE descriptor see [arXiv:2405.08137](https://arxiv.org/abs/2405.08137)


PANNA is a package to train and validate machine learned interatomic potentials based on different atomic environement descriptors (BP[1], modified-BP[2], LATTE) and atomic multilayer perceptrons.

* Tutorials and Documentation: [https://pannadevs.gitlab.io/pannadoc/](https://pannadevs.gitlab.io/pannadoc/)
* Mailing list: [https://groups.google.com/d/forum/pannausers](https://groups.google.com/d/forum/pannausers)
* Source: [https://gitlab.com/pannadevs/panna](https://gitlab.com/pannadevs/panna)
* Bug reports: [https://gitlab.com/pannadevs/panna/issues](https://gitlab.com/pannadevs/panna/issues)


#### It provides:

* input creation and packaging tools (for precomputed descriptors)
* network training tools in TensorFlow2 and JAX
* network validation tools
* LAMMPS plugins for CPU and GPU
* ASE plugins
* JAX-MD plugins

#### Installation:

PANNA is based on TensorFlow and JAX: the original training code is based purely on TF, while the new training code supporting the LATTE descriptor is based on JAX, only using TF for data processing.

To install the needed packages for the TF version please install through
```
    pip install .[TF]
```
To install the needed packages for the JAX version (uses TF-cpu, JAX-GPU), please install through
```
    pip install .[JAX]
```
To manage your own back-ends, just install the main package, with no tags.

If you want to use an older version (1.xx) of TensorFlow, please switch to the branch tagged 1.xx.


#### Tools and interfaces

PANNA comes with several tools to parse and process data.
Tools can be found in the `src/panna/tools` directory, and are described in the relative tutorial:
```
   https://pannadevs.gitlab.io/pannadoc/tutorials/Tools.html
```

PANNA potentials can be used to run MD in lammps through the plugin found in `src/panna/interfaces/lammps`.
The JAX version uses a different plugin, optimized for GPU through KOKKOS, found in `src/panna/interfaces/lammps_jax`.

The potentials can also be used in ASE through the calculators at `src/panna/interfaces/ASE` and `src/panna/interfaces/jax_ASE`.

PANNA JAX can be used in JAX-MD through the code found in `src/panna/interfaces/jaxMD`.

PANNA TF can interface with several MD packages via KIM project [3] model driver: MD_805652781592_000.

#### Testing:

Simple tests to check minimal functionality (TF2) can be run with:
```
    python3 -m unittest
```

from within the src/panna directory.
This command runs the TF2 tests for the following scripts in various conditions

```
    gvect_calculator.py
    tfr_packer.py
    train.py
    evaluate.py
```


#### REFERENCES

    [1] J. Behler and M. Parrinello; Generalized Neural-Network
    Representation  of  High-Dimensional  Potential-Energy
    Surfaces; Phys. Rev. Lett. 98, 146401 (2007)
    [2] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg;
    ANI-1: An extensible neural network potential with DFT accuracy
    at force field computational cost; Chemical Science,(2017), DOI: 10.1039/C6SC05720A
    [3] E. B. Tadmor, R. S. Elliott, J. P. Sethna, R. E. Miller and C. A. Becker;
    The Potential of Atomistic Simulations and the Knowledgebase of Interatomic Models.
    JOM, 63, 17 (2011)
