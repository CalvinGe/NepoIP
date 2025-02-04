# OpenMM-NepoIP

OpenMM-NepoIP is an advanced interface for using the NepoIP model in OpenMM simulations. It is implemented based on the [OpenMM-ML API](https://github.com/openmm/openmm-ml) with significant new features.

## Usage

To use the NepoIP model, we need to create a MLPotential object:

```python
from openmmml import MLPotential
potential = MLPotential('nepoip',model_path='your/path.pth', distance_to_nm=A_to_nm, energy_to_kJ_per_mol=kcal_to_kJ_per_mol)
```

Since the prediction of NepoIP is the difference between the QM and MM energy and forces, we need to add its prediction on the top of an MM system and specify which atoms should be considered the QM region (ML region).:

```python
from openmm.app import *
pdb = PDBFile('example_run/amber_md_frame1.pdb')
prmtop = AmberPrmtopFile('example_run/ala_amber.prmtop')

chains = list(pdb.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]

mm_system = prmtop.createSystem(nonbondedMethod=Ewald, nonbondedCutoff=0.9*nanometers)

ml_system = potential.createEmbedSystem(pdb.topology, mm_system, ml_atoms)
```

## Optimizing the speed

The computation of the reciprocal electrostatic potential in a periodic system can be time consuming and is currently parallelized in the `parallel_compute` function of `NepoIP/openmm-nepoip/openmmml/models/utils.py`:

```python
rec_sum = parallel_compute(kmax, cell, positions - point, charges, alpha, volume, 256)
```

The last argument of  `parallel_compute` is the number of wave vectors to compute in a batch. It should be adjusted depending on the hardware to optimize the simulation speed.