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
---
### Pre-processing
---
### Post-processing
---
### Misc