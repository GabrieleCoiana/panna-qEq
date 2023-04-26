# This is a minimal example of how the ASE plugin can be called to perform MD
# Please provide you own configuration and PANNA potential (extracted with extract_weights.py),
# or use the ones in doc/tutorial/tutorial_data/C_ASE/

import ase
import ase.md
from panna.interfaces.ASE import PANNACalculator

myconf = ase.io.read('myconf.xyz', ':')[0]
pcalc = PANNACalculator(config='PANNA_potential.in')
myconf.set_calculator(pcalc)
dyn = ase.md.langevin.Langevin(myconf, ase.units.fs, temperature_K=300, friction=2e-3)
dyn.run(100)

