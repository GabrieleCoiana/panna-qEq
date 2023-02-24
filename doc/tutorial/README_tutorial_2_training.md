## PANNA training

### Introduction

In the following tutorial, you will find a simple walkthrough of the main features of the code related to training, validating and deploying a network.
We will guide you through a series of examples on the provided sample dataset, presenting the main keywords of the input files (please check the documentation for a more comprehensive list).

---

### Data

We start with pre-prepared dataset files of atomic positions and corresponding total energy [1]. 

In the folder `tutorial_data` you will find 
10 files `train-*-10.tfrecord` distributed to two folders called `train` and `validate`. 
These files with .tfrecord extension are TensorFlow data files containing simulation data of `H2O`, `NH3` and `CH4`
molecules. For each molecule we have 300 different configurations therefore total data is made up of 900 elements.
Each .tfrecord file has 90 of these in a random order.

In this tutorial we use 80% of the data for training but we encourage you to experiment
with different distribution of data between training and validation purposes. 

---

### Input

We now train a small neural network by using the script `train.py` found in `panna` directory
with the input `train1.ini` found in the tutorial directory.

Let us have a look at the input: 
there are 5 sections in this input file and we will go through some of the essential input parameters.

##### [IO_INFORMATION]
Specify input and output directory addresses, as well as logging and saving frequency.

* `data_dir` -- The directory containing the `tfrecord` data files we want to use for training.
* `train_dir` -- The directory that will contain the training output and relevant meta-data from training.
* `log_frequency` -- Frequency (in steps) for logging the reached accuracy and status of the training. It should be frequent, but can lead to large file sizes.
* `save_checkpoint_steps` -- Frequency (in steps) of saving "checkpoint" or restart files for the training. 
This can be kept bigger, i.e. less frequent for production runs.

##### [DATA_INFORMATION]
Parameters about the descriptor functions we have employed 
to create the input.
Do not change these as long as you are using the tutorial data set.

* `atomic_sequence` -- If the input requires a particular atomic sequence to be preserved, 
as in Behler-Parinello vector, the atomic types are given here in the order employed 
for the construction of the descriptors (as a comma separated list of strings).
* `output_offset` -- The zero reference (such as energy reference) of each species 
(in the order specified) 
which will be used to offset the output trained on the network, leading to a faster 
overall training.

##### [TRAINING_PARAMETERS]
Specify the parameters for the training:

* `batch_size` -- The size of the mini batch, i.e. the number of examples we consider in each optimization step during training
* `learning_rate` -- The learning rate of the optimizer (default is constant). 
Bigger it is, bigger the steps taken in the parameter space during optimization. 
* `max_steps` -- The total number of optimization steps in the training as a stopping criteria.
Note that in the case of a restart, the last step count is preserved, and the new training starts 
from where the last one has finished, and would stop when the `max_steps` is reached.  

##### [DEFAULT_NETWORK]
The structure of the neural network is specified here.
At this step of the tutorial, we demonstrate how to use a common structure for all species.
We will later see how to specify different architectures for different species.

* `g_size` -- The length of atomic input array 
* `architecture` -- A string specifying the size of each hidden layer, plus the output layer (1). 
In our case for example `128:32:1` will lead to two hidden layers after the input 
(each fully connected with Gaussian activation)
and the output layer with linear activation, for a total of 3 sets of trainable weights and biases.
(see doc/PANNA_documentation.md for the formulas used.)
* `trainable` -- flags indicating whether each layer should be allowed to train or be kept frozen. In our case `1:1:1` means that all layers will be trained.
* `activations` -- flag specifying which activation to use for each layer, between linear (0), gaussian (1), radial basis function (2), rectifying linear unit (ReLU, 3), or hyperbolic tangent (tanh, 4). Default is gaussian activation for hidden layers and linear for the last layer, so we can omit the flag here or equivalently pass the string `1:1:0`.

---
### Training 

For tutorial purposes the parameters in `train1.ini` are for a reasonably short training.
To run the code you should `train.py` with your version of python3 and  passing the input file as a `config` argument, e.g.:

```
python3 ../../src/panna/train.py --config input_files/train1.ini
```

The run should take from a few seconds to a few minutes, depending on your machine.
The error with respect to target energy is written in `delta_e.dat` file at every `log_frequency` steps, for every element of the mini batch.
The mean error, mean absolute error, and standard deviation of error is written in `delta_e_std.dat`.
A copy of the code output is also saved in a `.log` file.
The more detailed results about the network can be inspected by using tensorboard, the data visualizer of TensorFlow.
To start tensorboard you should set the `logdir` argument of tensorboard to the `train_dir` of PANNA, e.g.:
```
tensorboard --logdir=./tutorial_train --host=0.0.0.0 --port=6006
```

(where the host address and port utilized can be chosen by the user).
While this command is running, we can visit the address above in a browser to visually inspect our training.

Initially let us concentrate on a few aspects: in the `SCALARS` tab, we first see the evolution of loss function
and evolution of error (Delta E) during the training. They should both rapidly decrease as the training starts. 
The standard deviation of the resulting error after 1000 optimization steps is 0.05 eV per simulation in our references, 
not an impressive result, but not bad for a few seconds of training on a simple network,
and very limited data.
Your result may vary slightly to reflect the randomization of weights, stochasticity in mini batch creation and training.

Another important scalar to check is the average energy contribution of each species 
to the total energy and its standard deviation.
Note that if the standard deviation of contributed energy per atom is very small, 
the network for that species contributes an almost constant value to the 
total energy across different batches. 
This means that the network has not yet learned to differentiate between different environments 
and would not represent a good potential. 

Secondly, we can check the `DISTRIBUTIONS` tab to have a 
bird's eye view of the evolution of all parameters for each species and layer, which is a good indicator of the status of the training.
For example, for each layer you can see the weights and biases in the network and how they have evolved during the training.
Networks tend to start learning from the layers closest to the output. 
You can observe this behavior in this very short training session by noticing that the weights and biases of the layers close to output 
are more structured and differentiated while the weights and biases close to input layer resembles very closely to the initial values.
And lastly if you are using recent versions of tensorboard, you can use the `PROJECTOR` tab to apply
dimensionality reduction and analysis tools on the different tensors of the network, such as PCA, t-SNE.

Notice that if we wanted a better result, we could just restart the training and allow for a longer run.
To try this, change the value of `max_steps` in the input file to a larger value and re-run the python directive as before. 
As long as the `train_dir` variable is the same as before, the code keeps track of the number of steps performed.
That way when you run the same command, the training will seamlessly resume,
and stop at the new `max_steps` steps. **Important** the delta_e.dat and delta_e_std.dat files will be overwritten. 
Try increasing the max_steps to 5000 (4000 more steps) and see whether you obtain lower error.
In our reference calculation, the std error reduced to 0.03eV while due to the small batch size fluctuations are present.

---
### Validation

The errors we examined so far are with respect to the training data.
To have a more accurate idea on the prediction quality of the network, 
we will now use the data in the `validate` folder, and calculate the network prediction by evaluating the network function on this data. 
In order to do that we use `evaluate.py` script in the `panna` folder and `validation1.ini` inside `input_files` as input file.  
The input has 3 sections:

##### [IO_INFORMATION]
Specifies the input and output folders:

* `data_dir` -- The folder containing the examples we want to evaluate the network on.
* `train_dir` -- The folder containing the output of the previous training to launch the trained network.
* `eval_dir` -- The folder that will contain the results of the network evaluation.

##### [TFR_STRUCTURE]
Here we need to specify some characteristics of the data inputs for validation. For the basic case of this tutorial, it is sufficient to specify the following:

* `g_size` -- The length of atomic input array

##### [VALIDATION_OPTIONS]
There are a few options on how to compute the validation. For tutorial purposes we will only specify the following

* `single_step` -- If `true`, evaluate only the last network found in the `train_dir`, if `false` all the checkpoints are evaluated.

With this simple input, we can now run the code like we did for training, e.g.:
```
python3 ../../src/panna/evaluate.py --config input_files/validation1.ini
```
This should very quickly compute the network on our evaluation set, 
generating a file with `.dat` extension in the folder specified in `eval_dir`. 
The name of the file indicates the checkpoint, the global step it imported the weights and biases from.
If you have been following the tutorial so far, it is step 5000, hence the file is `5000.dat`.
In this file, we can read the number of atoms at each simulation we used for validation, the DFT energy and the estimate of our neural network: 
we should find that, while very far from a useful result, also on the evaluation set the network starts to approximate the desired energy.
In our sample run, we have obtained 0.07eV as the standard deviation of the error. Your result may vary to indicate difference in the training.

We will now see how to extract the parameters of this model, 
to be reused in some external code for analysis, or to be imported in a new network for further training.

---
### Extracting weights 

We will now use the `extract_weights.py` code to save all parameters of the network: 
this code will take a checkpoint and generate two output files per layer per species, 
one with the weights and one with the biases. 
It will also generate a `.json` file specifying the structure of the network in human-readable format.

A simple input file for this code uses the following keywords:
##### [IO_INFORMATION]
* `train_dir` -- The training directory, containing the checkpoints.
* `output_dir` -- The name of the directory where the network will be saved.
* `step_number` -- The number of checkpoint for which we want to save the network.
* `output_type` -- `PANNA` (default value) generates network files in the described format, suitable for restart. Other formats are possible (e.g. `LAMMPS` is available).
* `gvector_input` -- Optional name of the input file used for creating the gvector descriptors. If provided, extra information is included in the output metadata file.

If we used the tutorial inputs, we can save the network by running:
```
python3 ../../src/panna/extract_weights.py --config input_files/extract_weights.ini
```
(this will by default save the weights of step 1000, please change the `step_number` variable if you have trained your network for longer).
A new folder (`saved_weights` in this case) will be created containing the network structure as `networks_metadata.json` and one `.npy` file for the weights and biases of each layer. 
These files are simply numpy arrays containing the vector or matrix in question, should one need to access them to import the parameters directly to another program.

---
### Modifying a network

We will now see how to use the saved weights as starting values for a new network, 
and how to create a more complex species specific network with specific trainable layers.

To do this, we will go back to the main training code `train.py` with a more complex input file, as the example provided in `train2.ini`. 
These are the new sections and keywords we will use:

#### [DEFAULT_NETWORK]

* `networks_metadata` -- This indicates the folder where the `networks_metadata.json` file containing the structure of the network is located. 

Let us now use the network we have previously trained as the default network. 
If we want to change something for a particular species, we can specify it in each atomic section.

#### [X]
Where `X` is an element type as specified in the `atomic_sequence`, eg. [H].

* `architecture` -- Behaves like the same tag in `[DEFAULT_NETWORK]`. Only needs to be specified if we want to change the architecture for this specific element.
* `trainable` -- Behaves like the same tag in `[DEFAULT_NETWORK]`. Only needs to be specified if we want to change the trainable flag for this specific element.
* `behavior` -- A colon separated string specifying where to get the weights for each layer, e.g. `load:new:new`. Accepts keywords `new` (start from random weights), `load` (load from file, see later), `keep` (keep the weights from the last checkpoint). 
* `layerX_w_file` -- Indicates the file to load the starting weights for layer number `X` (0 based index) of this species network. 
Only needs to be specified if different from the default network (in this case, what we are importing from the previous training session), 
and needs to be of the appropriate size.
* `layerX_b_file` -- Same as the previous, but for the biases.

To give a practical example of how these keywords can be used, let us imagine, for instance, that we want to modify the network we just trained in the following way:

* **N** -- We would like to resume the training on nitrogen where we left off, with no change. As this is the standard behavior, no new section is required.
* **O** -- We like the final result for oxygen and would like to freeze it as it is, with no further training. 
We will therefore add the `[O]` section and only specify the `trainable` field to indicate we do not want to train.
* **C** -- We want to retrain carbon, with the same geometry, but we would like to borrow the first layer from the one of oxygen, 
(you can imagine a scenario where we are copying from another carbon from a different training),
while starting the others from scratch. 
To specify this, we will need to indicate a file for each weight and bias element we import, using the keyword `new` for second and third layer.
* **H** -- Since hydrogen is present in all molecules, possibly with a variety of atomic environments, 
let us imagine that we would like to give it more freedom by adding another hidden layer before the output: 
we will restart the first layer (which has the same shape) from the previous training, 
while the others will be started from scratch. 
To achieve this, we will use the `architecture` keyword to indicate the new architecture. 
We can also use the `trainable` keyword to make all layers trainable. 
We do not need to add a `layer0_w_file` keyword, as this will be imported from the default behavior, 
but we will specify `new` for all other layers as they need to be reinitialized.

All in all, this section of our input file will therefore look like this (see `train2.ini`):
```
[DEFAULT_NETWORK]
g_size = 384
architecture = 128:32:1
trainable = 1:1:1
networks_metadata = saved_weights
[O]
trainable = 0:0:0
[C]
behavior = load:new:new
layer0_w_file = saved_weights/species_O_layer_0_weights_384x128.npy
layer0_b_file = saved_weights/species_O_layer_0_biases_128.npy
[H]
architecture = 128:32:32:1
trainable = 1:1:1:1
behavior = keep:new:new:new
```

As we can see we have complete control on the geometry and starting state of our network. 
Notice that this simulation should have a different `train_dir`, as it is not a restart but 
should be considered a new instance of training.

We can now run this new training again as 
```
python3 ../../src/panna/train.py --config input_files/train2.ini
```
and analyze our results in tensorboard by pointing it to the new training directory:
```
tensorboard --logdir=./tutorial_train_2 --host=0.0.0.0 --port=6007
```
We can see that, with all the changes made, this network may still be far from convergence, 
but by looking in the `DISTRIBUTIONS` section we can notice how some species are not training, 
some have restarted from scratch and others just resume from the previous state, just as we specified.

---
### A more realistic training scenario

We will now add more details on how to chose parameters to perform a more complete training for a realistic dataset.

### Learning rate
It may be desirable to decrease the learning rate during training, to mimic annealing. 
PANNA allows for an exponentially decaying learning rate following the formula,
other decay functions will be implemented in the future.
```
decayed_learning_rate = learning_rate * decay_factor ^ (global_step / decay_steps)
```
This can be obtaining with the addition of the following flags to the `[TRAINING_PARAMETERS]` section:

* `learning_rate_constant` -- When set to false, the learning rate decays exponentially according to the equation specified above.
* `learning_rate_decay_factor` -- The base `decay_factor` of the exponential decay.
* `learning_rate_decay_step` -- The timescale of the decay `decay_steps` in units of global steps.

### Regularization
Another useful option to control the training is the addition of a penalty 
in the cost function proportional to the sum of the parameters.
PANNA allows for two different forms of regularization for the moment: 
the sum of the weights squared (so called L2 regularization) 
or the sum of the absolute value of the weights (L1 regularization).
These can be applied with the addition of the following flags to the `[TRAINING_PARAMETERS]` section:

* `wscale_l1` -- Prefactor of the L1 regularization for the weights.
* `bscale_l1` -- Prefactor of the L1 regularization for the biases.
* `wscale_l2` -- Prefactor of the L2 regularization for the weights.
* `bscale_l2` -- Prefactor of the L2 regularization for the biases.

Another form of regularization is clipping the gradients. You can achieve this with the following
flag in the same section:

* `clip_value` -- the gradients will be clipped by value such that they will be in (-clip_value, +clip_value) range.
  a value of 0.0 will turn off this feature. Default is 0.0.

### Parallelization
PANNA also offers different flags to improve the algorithm parallelization for different architectures.
The following parameters can be specified in section `[PARALLELIZATION]` of the input file:

* `num_parallel_readers` -- number of parallel readers to read the tfrecord to buffer. Default is 1.
* `num_parallel_calls` -- number of parallel parsers that parse the tfrecord and load its content. Default is 1.
* `shuffle_buffer_size_multiplier` -- how many multiples of the batch-size of data 
should be shuffled before being used in batch creation. 
Default is 10.
* `prefetch_buffer_size_multiplier` -- how many multiples of batch-size of data should be read to buffer. 
* `inter_op_parallelism_threads` -- How many operations of the graph can be launched in parallel. See TF documentation for details. 
* `intra_op_parallelism_threads` -- How many threads are available for execution of each operation. See TF documentation for details. 

Note that inter and intra _op_parallelism settings can greatly affect the performance of the code 
but their optimization can be tedious as the optimum values vary between computing platform, and between the network architecture.


### Output options
Lastly, besides the already mentioned `log_frequency` and `save_checkpoint_steps`, 
two other keywords in the `[IO_INFORMATION]` section can help get better control on the log and saved output:

* `tensorboard_log_frequency` -- Specifies the frequency (in steps) 
for saving points for tensorboard visualization (if missing defaults to `log_frequency` value).
* `max_ckpt_to_keep` -- Specifies a maximum number of checkpoints to be saved, after which the oldest ones will be deleted.

---
### Training with forces
If force data are available for the input simulations and the derivatives of the descriptors have been precalculated, it is possible to modify the training loss so that forces are predicted and compared to the reference to drive the weights optimization.
The first step to achieve this is to compute and pack the derivatives of the descriptors, as described in Tutorial 1. We will here assume that the input TFRs contain the desired derivatives and to begin with we will consider them to be stored in the dense format (default).

To perform a training with forces cost, it is sufficient to specify a `forces_cost` different from 0 in the [TRAINING_PARAMETERS] card of the input file.
Please note that forces training tends to take more time and memory per step than normal energy training, but each step contains much more information, so that training parameters will have to be reoptimized.

To calculate forces in validation, it is only necessary to have precalculated the descriptors derivatives. To compute forces in validation, simply add the key `compute_forces = True` to the [VALIDATION_OPTIONS] card.
A new output file for each checkpoint with postfix `_forces.dat` will be created, containing for each example and atom one line with predicted forces (x, y, z) and, if available, reference forces.

#### Sparse derivatives
If the simulations consist of cells much larger than the cutoff, or of a lot of species, it might be beneficial to store the descriptors derivatives in a sparse format. The procedure to obtain this is specified in Tutorial 1, we will here list the keywords needed to benefit from this structure.

In training, besides the aforementioned `forces_cost`, the keyword `sparse_derivatives = True` will have to be added to the [DATA_INFORMATION] card.

Similarly in validation, besided requiring forces, the keyword `sparse_derivatives = True` will have to be added to the [TFR_STRUCTURE] card.

---
### REFERENCES

    [1] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg.
    ANI-1, A data set of 20 million calculated off-equilibrium conformations
    for organic molecules. Scientific Data, 4 (2017), Article number: 170193,
    DOI: 10.1038/sdata.2017.193
