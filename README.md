## PANNA
### Properties from Artificial Neural Network Architectures

### For the general structure (PANNA 1.0) see publication at: [Comp Phys Comm, 256, 107402 (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010465520301843)
### or the pre-review version at: [arXiv:1907.03055](https://arxiv.org/abs/1907.03055)
### For the latest version (PANNA 2.0) see publication in: [J. Chem. Phys. 159, 084117 (2023)](https://pubs.aip.org/aip/jcp/article/159/8/084117/2908459/PANNA-2-0-Efficient-neural-network-interatomic)
### or the pre-review version at: [arXiv:2305.11805](https://arxiv.org/abs/2305.11805)

PANNA is a package to train and validate all-to-all connected network models for BP[1] and modified-BP[2] type local atomic environment descriptors and atomic potentials.

* Tutorial and Documentation: [https://pannadevs.gitlab.io/pannadoc/](https://pannadevs.gitlab.io/pannadoc/)
* Mailing list: [https://groups.google.com/d/forum/pannausers](https://groups.google.com/d/forum/pannausers)
* Source: [https://gitlab.com/pannadevs/panna](https://gitlab.com/pannadevs/panna)
* Bug reports: [https://gitlab.com/pannadevs/panna/issues](https://gitlab.com/pannadevs/panna/issues)


#### It provides:

* an input creation tool (atomistic calculation result -> G-vector )
* an input packaging tool for quick processing of TensorFlow ( G-vector -> TFData bundle)
* a network training tool
* a network validation tool
* a LAMMPS plugin
* an ASE plugin


#### Testing:

Simple tests to check functionality can be run with:
```
    python3 -m unittest
```

from within the src/panna directory.
This command runs the tests for the following scripts in various conditions

```
    gvect_calculator.py
    tfr_packer.py
    train.py
    evaluate.py
```

#### Installation:

PANNA is based on TensorFlow.
If you want to use an older version (1.xx) of TensorFlow, please switch to the branch tagged 1.xx.
The main branch (this code) is based on Tensorflow version 2.xx (latest versions should have the best compatibility).

PANNA can be installed through package managers such as pip, e.g.
```
    pip install .
```
from the root package directory.

#### Tools and interfaces

PANNA comes with several tool to parse and process data.
Tools can be found in the `src/panna/tools` directory, and are described in the relative tutorial:
```
   https://pannadevs.gitlab.io/pannadoc/tutorials/Tools.html
```

PANNA potentials can be used to run MD in lammps through the plugin found in `src/panna/interfaces/lammps`.

They can also be used in ASE through the calculator at `src/panna/interfaces/ASE`.

Alternatively, PANNA can interface with several MD packages via KIM project [3] model driver: MD_805652781592_000.


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
