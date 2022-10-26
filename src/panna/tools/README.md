# HOW TO:
Currently PANNA is not distributed as a library. 
To use the tools, add PANNA folder to the shell variable PYTHONPATH

```script
   export PYTHONPATH=$PYTHONPATH:/path/to/panna
```

# binary_updater.py
This tool can be used to update old binary files containing the g\_vector to
the new format.

**Attention!** this tool have some hard coded global variables that need to
be changed

supported conversion:
  - no\_version to v0
  - no\_version to v0\_sparse
parameters:
  - -i or --indir: input directory
  - -o or --outdir: output directory
  - -p or --parallel: number of processes (default 2)


# ijk.py
This tool shows a heat map of the distribution of the G-vector elements.
parameters
  - -s or --source: source folder where examples files are located
  - -r or --r_cut: radial cutoff to plot in Angstrom
  - -a or --atomic_sequence: atomic sequence, comma separated
  - -b or --bins: number of bins on radial axis, angular axis will be generated
                  accordingly. optional, default: 100
  - -n or --number_of_elements: number of elements to use to create the graph
                                useful if the data set is too big.
                                optional, default: all the elements
  - --log : put the color scale of the heat map in log scale
  - --dump : dump the data used to generate the heat map in to a readable format

example of usage:
```script
python3 tools/ijk.py  -s path/to/the/dataset -r 5.0 -a H,N,C,O -n 100
```
