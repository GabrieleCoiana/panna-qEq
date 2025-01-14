## PANNA JAX

### Introduction

In the following tutorial, you will find a simple walkthrough of the main features of the JAX version of PANNA, including the LATTE descriptor, training, validating and deploying a network.
We will guide you through a series of examples on the provided sample dataset, presenting the main keywords of the input files (please check the documentation for a more comprehensive list).

---

### Data

For this tutorial, we use very few simple configurations of small organic molecules containing `H`, `C`, `O`, and `N`, that you will find in the folder `simulations` in `.example` format.
These are simply meant to run the tutorial very quickly and are by no means realistic.

If you want to run the tutorial on your data, please refer to the documentation for the `.example` json format, and change the input file accordingly.

---

### Input

We will train a small neural network by using the script `train_jax.py` found in the `src/panna` directory with the input `train_jax.ini` found in the `tutorial/input_files` directory.

Let us have a look at the input: there are 4 sections in this input file and we will go through some of the essential input parameters.

##### [IO_INFORMATION]
Specify input and output directories, and other information.

* `train_dir` -- The directory that will contain the training output and relevant meta-data from training.
* `data_dir` -- The directory containing the data files we want to use for training.
* `val_data_dir` -- The directory containing the data files we want to use for validation at the end of each epoch.
* `load_weights` -- Whether the network should be loaded at the beginning of training. A `False` value means the training will start from scratch. This is useful in the case of a restart, in which case setting this to true will start the training from the last available checkpoint. See the input files documentation for more options.
* `metrics` -- Which metrics to show while training, and on the validation set at the end of each epoch.

##### [DATA_INFORMATION]
Parameters about the input data.

* `input_format` -- The format of the input configuration data. Our json format is called `example`, see the input files documentation for more options.
* `atomic_sequence` -- The atomic symbols that will be found in the training data, as a comma separated list of strings.
* `output_offset` -- The zero energy reference of each species (in the order specified by the `atomic_sequence`) which will be used to offset the output trained on the network, leading to a faster overall training.

##### [MODEL_INFORMATION]
The structure of the neural network and the environment descriptor are specified here.
See the official paper for more details about the LATTE descriptor.

* `model_type` -- The model we want to use, for now `MLP` (MultiLayer Perceptron) is the only available option.
* `architecture` -- A string specifying the size of each hidden layer, plus the output layer (1). Layers sizes are separated by a colon: e.g. in this case `64:32:1` will lead to two hidden layers after the input (each with nonlinear activation) plus the output layer with linear activation, for a total of 3 sets of trainable weights and biases.
* `mlp_sp_weights` -- A list of colon separated letters indicating whether each layer should have specific weights for each species (`s`), common or embedded weights. All specific, like in this case, is also the default. See the input files documentation for more details.
* `descriptor_type` -- The local environment descriptor to be used. For now `LATTE` is the only option.
* `Rc` -- The cutoff radius (in Angstrom) to consider for the local environment.
* `descriptor_shape` -- The elements to include in the LATTE descriptor. The structure is the following:
    * Each group of elements is separated by a colon, tokens within each group are separated by a comma
    * The first token identifies how many of this particular descriptor type should be used
    * The remaining tokens indicate the structure of the descriptor type: `-` is a special token for the 2 body descriptor (radial only). For any other descriptor type, a set of letters represents the indices for a single body. Indices are interchangeable, but each index should appear twice.
    * In this example, the string `10,-:10,i,i:10,i,j,ij` identifies: 10 2-body elements, 10 3-body elements contracting 2 vectors, and 10 4-body elements contracting two vectors to a tensor of order 2.
* `sig` -- The width of each radial function (in Angstrom).
* `learnsig` -- Whether the width should be learned. The example value `False` is the default, see the input files documentation for more details.
* `spec_weights` -- Whether the descriptor should have `specific` parameters for each species, or common\shared (see the input files documentation for more details).
* `at_mod` -- Since JAX needs to compile the code for each batch with a different number of atoms, this value allows the user to decide to which integer multiple the atoms in a batch will be padded. As an example, with the sample value `100`, a batch with 123 atoms would be padded to 200 atoms. Smaller values use les padding, but require more compiled versions of the code, larger values use more padding, but compile less. If too many of the early training steps are very long, try increasing this value.
* `pair_mod` -- The same as `at_mod`, but for the number of pairs in the model, i.e. the sum for each atom of the number of atoms within its cutoff. The same rules as `at_mod` apply.


##### [TRAINING_PARAMETERS]
Here we specify details about the training lenght and parameters.

* `batch_size` -- The size of the mini batch, i.e. the number of examples we consider in each optimization step during training
* `learning_rate` -- The learning rate of the optimizer (Adam). We can either specify a single value (constant), or a sequence of constant or decaying learing rates. In this case the string `const:1e-4:10,` `exp:1e-4:1e-2:10` indicates a constant learning rate 0.0001 for 10 steps, then decreasing from 1e-4 to 1e-6 in the next 10 steps. See the input files documentation for more details.
* `steps_per_epoch` -- The number of steps that will be considered an "epoch" of training. This does not necessarity need to correspond to the time to see all data once, but is the amount of steps after which the model will be saved and validation will be performed.
* `max_epochs` -- The total number of epochs to train for before stopping.
Note that in the case of a restart (with optimizer file), the last epoch count is preserved, and the new training starts from where the last one has finished, and would stop when the `max_epochs` total is reached.
* `l1_coef` -- The L1 regularization coefficient (of AdamW).
* `forces_cost` -- The cost of the forces component in the loss function. This should be greater than zero for forces target to be considered in training. See the input files documentation for more loss function options.

---
### Training 

For tutorial purposes the parameters in `train_jax.ini` are for an incredibly short training.
To run the code you should `train_jax.py` with your version of python, passing the input file as a `config` argument, e.g.:

```
python3 ../../src/panna/train_jax.py --config input_files/train_jax.ini
```
or
```
panna_jax --c input_files/train_jax.ini
```

The code should run in a few seconds, although the first steps might be particularly slow as the code needs to be compiled for a specific (padded) batch size.
As the training progresses, you can see the instantaneous loss and training errors (RMSE and MAE over energy and forces, averaged over a few batches just to limit the variability).
At the end of an epoch (only 5 steps in our case), the resulting model is used over the whole validation set, and the corresponding figures of merit are reported.

At the end of training, a folder `train_jax` contains all the outputs of the run: the file `metrics.dat` containing training and validation summaries for each epoch, and the folder `model` containing the weights at the end of each epoch, and the optimizer state at the end of the run.

---
### Inference

After training, we can evaluate our models on some new data, either to obtain more detailed test figures of merit, or simply to perform new predictions (we will see how to use external simulation engines in the next section).

To perform a simple evaluation, we can just use the same input file, but adding a couple of keywords to the `[IO_INFORMATION]` section:

* `task` -- Setting this to `val` will enable validation mode.
* `val_dir` -- The folder that will contain the results of the network evaluation.

For simplicity, we have copied an input modified with these keywords as `val_jax.ini`. We can now run exactly the same code as:

```
python3 ../../src/panna/train_jax.py --config input_files/val_jax.ini
```
or
```
panna_jax --c input_files/val_jax.ini
```

This will quickly evaluate all the files in `val_data_dir` and create two output files in `val_jax` for each network in the `models` folder: one with the name of the network `.dat` and the second ending in `_forces.dat`.
Inspecting the files, we can see that the former contains one entry per structure with the predicted and reference energies, while the latter contains one line per atom per structure with predicted and reference force components.

---
### Exporting the potential

A model trained in PANNA_JAX can be used in several external simulation engines.

For python based simulators like ASE and JAXMD, we can directly use the original `.ini` file used to train the model and the `.pkl` file created during training. To make sure we load the right model in the `[IO_INFORMATION]` field we should set `load_weights` to `True` and `weights_file` to the path of the model we want to load.

For LAMMPS, we need to export the model to a different format instead, as specified below.

##### ASE

To use the potential with ASE, just import the calculator found in `src/panna/interfaces/jax_ASE.py`, you can then create a calculator by specifying the `.ini` file, something like:
```
from panna.interfaces.jax_ASE import PANNAJAXCalculator
calculator = PANNAJAXCalculator('path/to/train.ini')
```

##### JAXMD

To use the potential in JAXMD, we can use the neighbor list defined in `src/panna/interfaces/jaxMD.py`. We can then obtain neighbors, energy and forces functions by simply supplying the `.ini` file:

```
from panna.interfaces.jaxMD import PANNAJAX_neighbor_list
neighbor_fn, energy_fn, force_fn = PANNAJAX_neighbor_list(displacement_fn, 'path/to/train.ini', species, box)
```

##### LAMMPS

To use this potential in LAMMPS, we need to export it in a format that can be loaded from the C++ plugin, which needs to be installed in main code.
We refer to the `README` in `src/panna/interfaces/lammps_jax` for detailed instructions on how to compile the plugin, and will mention here only the main features of the extraction process.

The most performant PANNA plugin for LAMMPS, especially running on GPU, relies on a pyTorch version of the model for extraction, therefore pyTorch has to be installed in your environment to run this.

To extract the potential, we will once more reuse the original `.ini` file used for training, and just add a few extra keywords.
The essential keywords to add are:

* `extract_dir` -- The directory where we want to save the exported model.
* `extract_format` -- The format of the exported model: `torch` is the one with best performance.

Like the python formats, we should also make sure to enable `load_weights` and set `weights_file` to the path of the model we want to export.

We can then export through the `extract_jax.py` code:
```
python3 ../../src/panna/jax/extract_jax.py --config input_files/export_jax.ini
```

After running this, you should see a `lammps_model` folder containing two files: a human readable `panna.in` containing the essential parameters of the model, and a pyTorch file `model.pt` containing the weights.
Within LAMMPS, we could now load this model by specifying this folder as the parameter of a `latte` pair style.




