## PANNA
### Properties from Artificial Neural Network Architectures

### See publication at: [Comp Phys Comm, 256, 107402 (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010465520301843)
### or the pre-review version at: [arXiv:1907.03055](https://arxiv.org/abs/1907.03055)

PANNA is a package to train and validate all-to-all connected network models for BP[1] and modified-BP[2] type
local atomic environment descriptors and atomic potentials.

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


#### Testing:

Simple tests to check functionality can be run with:
```
    python3 -m unittest
``` 

from within the panna directory. 
This command runs the tests for the following scripts in various conditions

```
    ./panna/gvect_calculator.py 
    ./panna/tfr_packer.py 
    ./panna/train.py
    ./panna/evaluate.py 
```    


PANNA potentials can be used in several MD packages via KIM project [3] model driver:
MD_805652781592_000


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
