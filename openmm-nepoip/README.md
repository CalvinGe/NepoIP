# OpenMM-NepoIP

OpenMM-NepoIP is an advanced interface for using the NepoIP model in OpenMM simulations. It is implemented based on the [OpenMM-ML API](https://github.com/openmm/openmm-ml) with significant new features.

## Usage

To conduct NepoIP/MM simulation, we need to the following:

### 1. Create a MLPotential object:

```python
from openmmml import MLPotential
...
# distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# energy: model is in kcal/mol, OpenMM is in kJ/mol
kcal_to_kJ_per_mol = 4.184
...
potential = MLPotential('nepoip', model_path='../example_model/nepoip_dftb.pth',
                        distance_to_nm=A_to_nm,
                        energy_to_kJ_per_mol=kcal_to_kJ_per_mol,
                        cut_esp = True)
```

The key arguments for creating a MLPotential:

* name: available options are ['nepoip', 'nepoipd']. Use 'nepoip' for deployed **NepoIP** or **NepoIP-0** model, and use 'nepoipd' for the **NepoIP-d** model.

* model_path: the path for a deployed model

  >  Only models that are deployed successfully from `nequip-deploy` can be used

* cut_esp: whether to use a cutoff scheme for computing the external electrostatic potential on the ML region, otherwise, use the Ewald summation scheme as default. This will only be effective when the potential is added on a periodic OpenMM system.

### 2. Add the Potential on Our System

Since the prediction of NepoIP is the difference between the QM and MM energy, we need to add its prediction on the top of an MM system and specify which atoms should be considered the QM region (ML region).:

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

If the Ewald summation for computing electrostatic potential is used, the computation of the reciprocal components can be time consuming and is currently parallelized in the `parallel_compute` function of `NepoIP/openmm-nepoip/openmmml/models/electrostatic.py`:

```python
rec_sum = parallel_compute(kmax, cell, positions - point, charges, alpha, volume, 256)
```

The last argument of  `parallel_compute` is the number of wave vectors to compute in a batch. It should be adjusted depending on the hardware to optimize the simulation speed.