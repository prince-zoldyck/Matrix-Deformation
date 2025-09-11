# We will use the debye approximation QH class
from pymatgen.analysis.quasiharmonic import QuasiHarmonicDebyeApprox
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

import mace
from mace.calculators import MACECalculator
import numpy as np

# You will need to load a different MACE model path, see
# the mace tutorial for details
calc = MACECalculator(model_paths="/home/ttian7/Dropbox/Dev/MACE-models/MACE-MP0b/mace_agnesi_small.model",
                      # device="cuda" only works for Nvidia GPUs
                      # for mac you may just use device="cpu" or omit it
                      device="cuda")

# The best_a is determined from a previous example code
best_a = 47.05 ** (1/3)
cu = bulk("Cu", cubic=True, a=best_a)

orig_cell = cu.get_cell() 

deformation = [(np.eye(3)*((1+x)**(1.0/3.0))) for x in np.linspace(-0.07, 0.07, 20)]


energies = [] #Empty matrix to store energies
volumes = [] #Empty matrix to store volumes

for defo in deformation:
    new_cell = np.dot(orig_cell, defo)
    
    # Use the following 2 lines if you want to just scale the cell
    new_atoms = cu.copy()
    new_atoms.set_cell(new_cell, scale_atoms=True)

    # Do not forget to attach calculator to each Atoms instance!
    new_atoms.calc = calc
    new_energy = new_atoms.get_potential_energy()
    
    energies.append(new_energy)
    volumes.append(new_atoms.get_volume())

# The qh class will get you the gibbs free energy at each temperature point
qh = QuasiHarmonicDebyeApprox(energies=energies, volumes=volumes, structure=AseAtomsAdaptor.get_structure(cu), )

T = qh.temperatures[0]
G = qh.gibbs_free_energy[0]
cu.calc = calc
H = cu.get_potential_energy()
S = -(G - H)/T
# Are we calculating the H and S correctly?
print(T, G, H, S)
