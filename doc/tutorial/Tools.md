## PANNA Tools

The PANNA package come with many tools (found in the `tools` folder) to help interface with other programs and process input or output data.
In the following, we give a brief overview of most of the tools, divided by category.

Please be aware that while these tools have been used and should be functional, they are not guaranteed to be completely up to date with all the features of the code, and might require some tweaking to be adjusted to your specific needs.
We recommend looking at the source code and check some sample files to make sure the results are what you expect, and modify the tools as you see fit (for example, some common parameters might be hardcoded in the source file).

Please note that some of these tools might depend on other external libraries or packages.

---
### Input data parsers

In this category, we list all tools used to import data from other programs or datasets into `.example` files that can be used for training.

#### Quantum ESPRESSO parsers
This tool parses Quantum ESPRESSO outputs. It comes in two forms: the preferred one is a parser for the `.xml` output files, but we also provide a minimal parser for plain text outputs.

The `xml` output parser will look for all the files with `.xml` extension in a directory (and its subdirectories) and generate the corresponding PANNA `.example` files of the same name.
An optional arguement `--addhash` can be used to add a hash to the filename to avoid clashes.

It expects the input path to act on and output dir, e.g.:
```
python parse_qexml.py -i in_dir -o out_dir
```

The other option is a plain text parser: this tool operates on every file in a directory (expects all or them to be QE outputs) and generates a `.example` file. It is also capable of extracting different configurations from a relaxation calculation and create an example from each. Here too the flag `--addhash` can be used to add a hash to the filename.

A sample input is:
```
python qe_output_parser.py -i in_dir -o out_dir
```


#### Lammps parsers
This tool is intended to parse lammps simulations: one version can parse some common `dump` outputs (or can be modified for custom ones) while another parses `xyz` configuration files.

The dump parser tries to parse one or multiple dump files (in which case it looks for `.atom` files, or can be supplied a different extension through the `-ext` option).
It should recognize the most common keywords, but please check results on your sample files and modify as appropriate (for example, the tool assumes the units to be `metal`). It requires the `pandas` library.

A basic usage would look like:
```
python lammpsdump2json.py -dumpd in_dir -o out_dir
```

Alternatively, an `xyz` position file can be parsed for the configurations and a second file can be used to read the energies.

In this case, a sample usage would be:
```
python lammpsxyz2json.py -p positions.xyz -e energies.dat -o out_dir
```


#### Vasp parser
This tool parses a Vasp `.xml` output file into `.example` json files (one per configuration, see the code for more options).

Basic usage is:
```
python vaspxml_parser.py -i vasprun.xml -o out_dir
```


#### USPEX parser
This tool parses a USPEX `POSCAR` file for configurations and a second file for energies and creates multiple `.example` json files.

Basic usage is:
```
python uspex_poscar2json.py -p poscar -e energy.dat -o out_dir
```


#### ANI1 parser
This tool parses data from the ANI-1 dataset of small molecules (information about it can be found here `https://github.com/isayev/ANI1_dataset`).

The code requires the package `h5py` to read the data.
It expects a single HDF5 file as an input and the folder where to extract the examples.
The output folder will be structured like the dataset (one folder per molecule) and contain all the `.example` files with hashed names.
Be aware that extraction might take quite some time for the larger pieces of the dataset.

A sample usage would be:
```
python ani_parser.py -in ani_gdb_s01.h5 -out out_dir
```


#### Extended xyz parser
This tool parses multiple configurations from multiple "extended xyz" files: it crawls a directory for all `.*xyz` files and generates a `.example` for each structure. Like other tools, the keyword `--addhash` can be used to add a hash to the filename.
Please refer to the code for more details, and parameters that can be set for your specific usage.
This tool requires the `pandas` library.

A basic usage would look like:
```
python extxyz_parser.py -i in_dir -o out_dir
```


---
### Data exporters

Here we collect some tools that can be useful to convert your  `.example` files to other formats for further processing.

#### Exporter to Quantum ESPRESSO
This tool is supposed to help you create QE inputs from your examples. Of course QE inputs require many more parameters, therefore this script should be customized by the user to fill in the details as appropriate. Please inspect the code to see the provided sample input parameters. The cell and positions will be filled in from examples in a folder, and a new folder will be created for each structure with an input file and if needed a job file as well.

Basic usage:
```
python example2qe.py -i in_dir -o out_dir
```

#### Exporter to LAMMPS
This tool lets you export `.example` files as LAMMPS positions files. It will act on all the files in the input folder and create one folder per configuration containing the positions file.

A basic usage would look like:
```
python example2lammps_pos.py -i in_dir -o out_dir
```


#### Exporter to animated XCrySDen format (axsf)
This tool lets you export multiple `.example` files in a folder to a single XCrySDen animation `.axsf` file, for simple visualization of multiple configurations.

The basic usage is:
```
python json2axsf.py -i in_dir -o out_dir
```


---
### Pre-processing

In this section we include tools that can be useful to analyze your data before training. They are mostly tools that help you analyze the atomic environments and can guide you in the selection of the descriptor parameters.

#### Gvector visualizer
This simple tool allows you to visualize the radial and  angular bins of your descriptor, in order to check the width and distribution of the sampling bins. It directly takes as input the configuration file of `gvect_calculator`, and for now it works with mBP symmetry functions.
The tool requires the `matplotlib` package with a GUI backend to visualize the final plot.

A typical usage is:
```
python gvect_param_check.py -c gvect_input.ini
```

#### Compute descriptor statistics
This tool considers the descriptors of all atoms in all configurations in a folder and it computes the average and standard deviation for each bin.
The tool requires the `pandas` package. It expects binary descriptor files in the input folder and outputs a single `json` with the average and standard deviation lists.

A typical usage is:
```
python normalize_input_vectors.py -s in_dir -o out_dir
```

#### Visualize angular atomic distribution
This tool is used to analyze the distribution of atoms in the radial bins of the descriptor. It considers all configurations in a folder (`.example` files) and generates species resolved plots with the number of atoms that are found in a given angular and radial bin. Radial cutoff (in Angstrom) and atomic sequence (comma separated) need to be specified, see the source file for other plotting options (number of bins, log scale, saving the data of the plot).
The tool requires the `matplotlib` package with a GUI backend to visualize the final plot.

A basic usage would look like:
```
python ijk.py -s in_dir -r 5.0 -a H,N,C,O
```

#### Compute fingerprints and cluster
This is a set of tools that is used to compare configurations and possibly cluster them, in order to study the distribution of your training data. I comprises of 3 different codes that can be used in succession:

- `fingerprint_USPEX_erf.py` is used to compute the "fingerprint" descriptor for each structure and species pair as defined in `doi.org/10.1063/1.3079326`. Some of the fingerprint parameters are hardcoded and need to be changed in the source to obtain different fingerprints. Weights are also computed proportional to the number of atoms of a specific species pair in the structure. The script acts on all `.example` in the input directory and creates `.fprint` files in an output directory specified as:
```
        python fingerprint_USPEX_erf.py -i in_dir -o out_dir
```

- `fingerprint_distances.py` acts on fingerprint files and computes the distances between all configurations in a folder. Distance measure implemented are either the cosine distance from `doi.org/10.1063/1.3079326` or euclidean distance (flag `--euclidean`). It produces the distance matrix between the examples (as `.npy` upper triangular matrix) as well as a file with energy and volume data of all configurations. Input needs to specify the species to consider in computing the distance (as a comma separated sequence), in addition to input and output folders:
```
        python fingerprint_distances.py -i in_dir -o out_dir -s H,C,O
```

- `fingerprint_clusterization.py` finally starts from the distance matrix to perform a basic distance-based clusterization (creates clusters where all examples are separated by a distance larger than a given threshold). The script will cluster based on all distances between a minimum and maximum value (defaulting to min and max distances) sampled with a given step (starting at half the step). Optionally, energy can be considered in the clustering, see the source code for more details. The final ouput consists of a file `clusters_interval*.dat` with one row per threshold value containing the indices of the clusters for each example; and a file `clusters_probed*.dat` containing the used threshold and number of clusters for each line.
A sample input for clusterization (sampling thresholds 0.02, 0.04) would be:
```
        python fingerprint_clusterization.py -i indir -o outdir -min 0.01 -max 0.05 -dint 0.02
```


#### Estimate descriptor variance
This tool can be useful to understand how much variability is there the atomic environments for a given configuration. It considers all binary descriptor files in a folder, and for each configuration it computes the cosine distances between the descriptors of all pairs of atoms. It then outputs a text file with one line per configuration containing the filename, the energy and the variance of the distances vector.

The basic usage is:
```
python gvector_environmental_variance_estimate.py -i in_dir -o out_dir
```


---
### Post-processing

The tools included in this section can be used to process some of the data, outputs of training or validation for further uses.

#### binary_updater.py
This tool can convert between different formats of the binary gvectors. The currect version of gvector is called "v0", and the main usage of this tool is converting from the _dense_ to the _sparse_ format of derivatives. For back compatibility, the tool also supports conversions from the old "no version" encoding.

Some of the converions options (versions and data to be included) are hardcoded and need to be modified in the source file, while the input and output directory are specified as usual:
```
python binary_updater.py -i in_dir -o out_dir
```

#### checkpoint_to_csv.py
This tool acts on a folder of checkpoints generated during training and extracts some of the data displayed by Tensorboard as `.csv` text files.

Standard usage is:
```
python checkpoint_to_csv.py -i in_dir -o out_dir
```

#### gvector_writer.py
This tool can read gvector `.bin` binary files and convert them to plaintext. When passed a directory it will create a `*_plain.dat` file in the same folder for each configuration with the gvector of one atom per column. The argument `-a x` changes the behaviour to create a single file with the average gvector over all configurations for the first `x` elements (as specified at gvector creation time).

Standard usage is:
```
python gvector_writer.py -s in_dir
```

---
### Training procedures

#### iterative_magnitude_pruning.py
This tool is used to set up a training procedure known as iterative magnitude pruning (see https://arxiv.org/abs/1803.03635), consisting of retraining a network multiple times while progressively masking (i.e. eliminating) some of the weights to obtain a final model that is sparse and contains way fewer parameters.

To perform this multiple training, we need to set up our normal `train.ini` input, but then create another input file looking like this:
```
[IMP]
base_input = train.ini
prunable = 1:1:0
ratio = 0.3
N_iter = 10
start_step = 1
weights_step = 1000000

```
Where the `base_input` is our original input, the `prunable` flag indicates which layers will be masked, the `ratio` is the fraction of remaining weights to be removed at each iteration, `N_iter` is the total number of iterations, `start_step` the step to which we will reset the weights, and finally `weights_step` the step we will use to build the pruning mask.

With this input, the tool can simply be called as:
```
python iterative_magnitude_pruning.py -c IMP.ini
```
This will generate multiple folders, one for each pruning iteration, and progressively start the training in each of them. Inspecting the `train.ini` in each folder, you can see how the tool modifies the original input. Should the tool not finish running all iterations, it will be sufficient to run the training in each folder in sequence, to obtain the same final result.



