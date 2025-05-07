# import openmmtools
from openmm.app import *
import numpy as np
from openmmml import MLPotential

import sys
import openmm
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation, StateDataReporter, PDBReporter
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole, nanometers, bar

# load the alanine pdb
pdb = PDBFile('./amber_md_frame1.pdb')
prmtop = AmberPrmtopFile('./ala_amber.prmtop')

forcefield = ForceField('amber99sb.xml', 'tip3p.xml')

# pdb.topology.setPeriodicBoxVectors(None)
# mm_system = forcefield.createSystem(pdb.topology, nonbondedMethod=Ewald, nonbondedCutoff=0.9*nanometers, ewaldErrorTolerance=0.005, rigidWater=False)
mm_system = prmtop.createSystem(nonbondedMethod=Ewald, nonbondedCutoff=0.9*nanometers, ewaldErrorTolerance=0.000235, constraints=HBonds, rigidWater=False)

box_vectors = pdb.topology.getUnitCellDimensions()
# Set the box vectors in the system if they exist
if box_vectors is not None:
    a = [box_vectors[0], 0.0, 0.0] * nanometers
    b = [0.0, box_vectors[1], 0.0] * nanometers
    c = [0.0, 0.0, box_vectors[2]] * nanometers
    mm_system.setDefaultPeriodicBoxVectors(a, b, c)

# distance: model is in Angstrom, OpenMM is in nanometers
A_to_nm = 0.1
# energy: model is in kcal/mol, OpenMM is in kJ/mol
kcal_to_kJ_per_mol = 4.184
potential = MLPotential('nepoipd', model_path='../example_model/nepoipd_wB97x.pth',
                        distance_to_nm=A_to_nm,
                        energy_to_kJ_per_mol=kcal_to_kJ_per_mol,
                        cut_esp = False)

path = './'
# potential = MLPotential('ani2x')

chains = list(pdb.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]

# Create an integrator with a time step of 1 fs
temperature = 300.0 * kelvin
frictionCoeff = 1 / picosecond

ml_system = potential.createEmbedSystem(pdb.topology, mm_system, ml_atoms)
# ml_system = mm_system

integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, 
                                   0.002*picosecond)

# Setup the simulation
simulation = Simulation(pdb.topology, ml_system, integrator)
simulation.context.setPositions(pdb.positions)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--traj', type=str, help='parallel traj index')
args = parser.parse_args()

simulation.context.setVelocitiesToTemperature(temperature)

# Add a barostat for NPT simulation
barostat = openmm.MonteCarloBarostat(1.0*bar, temperature)
ml_system.addForce(barostat)
simulation.context.reinitialize(preserveState=True)
    
# Production MD in NPT
print('Running production MD in NPT...')
production_steps = 500  # Number of steps for production MD

for reporter in simulation.reporters:
        simulation.reporters.remove(reporter)

simulation.reporters.append(openmm.app.DCDReporter(path+'/trajectory_' + args.traj + '.dcd', 100))

simulation.reporters.append(openmm.app.StateDataReporter(path+'/ala_sol_2ns_md_'+args.traj+'.out', 100, step=True,
        potentialEnergy=True, temperature=True, progress=True, remainingTime=True, volume=True,
        speed=True, totalSteps=production_steps, separator='\t'))
simulation.step(production_steps)

simulation.saveCheckpoint(path+'/md_checkpoint_'+args.traj+'.chk')

print('MD Simulation finished.')
