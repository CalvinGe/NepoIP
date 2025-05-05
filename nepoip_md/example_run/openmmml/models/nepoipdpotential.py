"""
nequippotential.py: Implements the NepoIP potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from typing import Iterable, Optional, Union, Tuple
from openmmml.models.utils import simple_nl, get_kmax
from openmmml.models.electrostatic import ewald_esp, nopbc_esp
from openmm.unit import angstrom, nanometer
import numpy as np


class NepoIPdPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates NequipPotentialImpl objects."""

    def createImpl(self, name: str, model_path: str, distance_to_nm: float, energy_to_kJ_per_mol: float, cut_esp: bool=False, atom_types: Optional[Iterable[int]]=None, **args) -> MLPotentialImpl:
        return NepoIPdPotentialImpl(name, model_path, distance_to_nm, energy_to_kJ_per_mol, cut_esp, atom_types)

class NepoIPdPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the NepoIPd potential.

    The potential is implemented using NepoIPd to build a PyTorch model.
    A TorchForce is used to add it to the OpenMM System. Note that you must
    provide a deployed model. No general purpose model is available.

    There are three required keyword arguments

    model_path: str
        path to deployed NepoIP model
    distance_to_nm: float
        conversion constant between the nequip model distance units and OpenMM units (nm)
    energy_to_kJ_per_mol: float
        conversion constant between the nequip model energy units and OpenMM units (kJ/mol)
    cut_esp: bool
        whether to use a cutoff scheme for computing the electrostatic potential

    for example

    >>> potential = MLPotential('nepoipd', model_path='example_model_deployed.pth',
                        distance_to_nm=0.1, energy_to_kJ_per_mol=4.184,
                        cut_esp=True)    
    
    There is one optional keyword argument that lets you specify the nequip atom type of 
    each atom. Note that by default this potential uses the atomic number to map the NepoIP atom type. 
    This will work if you trained your NepoIP model using the standard `chemical_symbols` option. If you
    require more flexibility you can use the atom_types argument. It must be a list containing an 
    integer specifying the nequip atom type of each particle in the system.

    atom_types: List[int]


    """

    def __init__(self, name, model_path, distance_to_nm, energy_to_kJ_per_mol, cut_esp, atom_types):
        self.name = name
        self.model_path = model_path
        self.cut_esp = cut_esp
        self.atom_types = atom_types
        self.distance_to_nm = distance_to_nm
        self.energy_to_kJ_per_mol = energy_to_kJ_per_mol

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        

        import torch
        import openmmtorch
        import nequip._version
        import nequip.scripts.deploy

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        
        nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        charges = []

        for i in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge.value_in_unit(openmm.unit.elementary_charge))
            
        # Extract PME parameters
        for force in system.getForces():
            if isinstance(force, openmm.openmm.NonbondedForce):
                nonbonded_force = force
                break

        Boxvectors = topology.getPeriodicBoxVectors()
        
        if Boxvectors is not None:
            Boxvectors = Boxvectors.value_in_unit(angstrom)

            if nonbonded_force is not None:
                # ewald_tolerance = nonbonded_force.getEwaldErrorTolerance()
                ewald_tolerance = 0.005
                cutoff = nonbonded_force.getCutoffDistance().value_in_unit(angstrom)

                alpha = np.sqrt(-1*np.log(2*ewald_tolerance))/cutoff

                Boxvectors = torch.tensor(Boxvectors)
                kmax = get_kmax(Boxvectors, alpha, ewald_tolerance)

            else:
                raise ValueError("No NonbondedForce found in the system")

        residue_atoms = {}

        # Iterate over residues and store atom indices
        for res in topology.residues():
            atom_indices = [atom.index for atom in res.atoms()]
            residue_atoms[res.index] = atom_indices  # Using residue index as key


        class NepoIPdForce(torch.nn.Module):

            def __init__(self, model_path, includedAtoms, indices, periodic, cut_esp, distance_to_nm, energy_to_kJ_per_mol, charges, alpha=None, cutoff=None, kmax=None, atom_types=None, verbose=None):
                super(NepoIPdForce, self).__init__()
                
                # conversion constants 
                self.register_buffer('nm_to_distance', torch.tensor(1.0/distance_to_nm))
                self.register_buffer('distance_to_nm', torch.tensor(distance_to_nm))
                self.register_buffer('energy_to_kJ', torch.tensor(energy_to_kJ_per_mol))

                self.model, metadata = nequip.scripts.deploy.load_deployed_model(model_path, freeze=False)

                # self.default_dtype= {"float32": torch.float32, "float64": torch.float64}[metadata["default_dtype"]] # !!! modified from  metadata["model_dtype"]
                self.default_dtype=torch.float32
                torch.set_default_dtype(self.default_dtype)

                self.register_buffer('r_max', torch.tensor(float(metadata["r_max"])))
                
                if atom_types is not None: # use user set explicit atom types
                    nequip_types = atom_types
                
                else: # use openmm atomic symbols

                    type_names = str(metadata["type_names"]).split(" ")

                    type_name_to_type_index={ type_name : i for i,type_name in enumerate(type_names)}

                    nequip_types = [ type_name_to_type_index[atom.element.symbol] for atom in includedAtoms]
                
                atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]

                self.atomic_numbers = torch.nn.Parameter(torch.tensor(atomic_numbers, dtype=torch.long), requires_grad=False)
                #self.N = len(includedAtoms)
                self.atom_types = torch.nn.Parameter(torch.tensor(nequip_types, dtype=torch.long), requires_grad=False)

                self.cut_esp = cut_esp

                self.cutoff = cutoff
                self.alpha = alpha
                self.kmax = torch.tensor(kmax)

                if periodic:
                    self.pbc = torch.nn.Parameter(torch.tensor([True, True, True]), requires_grad=False)
                    self.calc_esp = ewald_esp
                else:
                    self.pbc = torch.nn.Parameter(torch.tensor([False, False, False]), requires_grad=False)
                    self.calc_esp = nopbc_esp

                self.charges = torch.tensor(charges)


                # indices for ML atoms in a mixed system
                if indices is None: # default all atoms are ML
                    self.indices = None
                else:
                    self.indices = torch.tensor(indices, dtype=torch.int64)

                
                self.residue_atoms = torch.zeros_like(self.charges)
                for res_idx, atom_indices in residue_atoms.items():
                    self.residue_atoms[atom_indices] = res_idx

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                
                # Created a padded forces tensor.     
                forces_padded = torch.zeros_like(positions).to(self.default_dtype)
                
                # setup positions
                positions = positions.to(dtype=self.default_dtype)
                charges = self.charges.to(dtype=self.default_dtype)
                charges = charges * 18.2223 # Convert to Amber's Unit such that E=q1*q2/r where E is in kcal/mol and r in Angstrom

                positions = positions*self.nm_to_distance
                # positions = torch.tensor(positions, requires_grad=True)

                ml_positions = positions[self.indices]
                ml_charges = charges[self.indices]
                    
                mask = torch.ones_like(charges, dtype=torch.bool)
                mask[self.indices] = False
                mm_positions = positions[mask]
                
                boxvectors = boxvectors if boxvectors is not None else torch.eye(3)

                box_vectors = boxvectors*self.nm_to_distance
                box_vectors = box_vectors.to(dtype=self.default_dtype)
                    
                if self.cut_esp:

                    esp = torch.zeros_like(ml_charges, dtype=self.default_dtype).to(positions.device)
                    grad_mm = torch.zeros((mm_positions.shape[0], ml_charges.shape[0], 3), 
                        dtype=self.default_dtype).to(positions.device)
                    elec_field = torch.zeros((ml_charges.shape[0], ml_charges.shape[0], 3), dtype=self.default_dtype).to(positions.device)
 
                    r = positions.unsqueeze(1).repeat(1, len(self.indices), 1) - positions[self.indices] 
                    for j in range(3):
                        box_length = box_vectors[j][j]
                        r[:, :, j] -= torch.round(r[:, :, j]/box_length) * box_length

                    distance = torch.sum(r ** 2, dim=2) ** 0.5
                    region_dist, _ = torch.min(distance, dim=1)

                    neighbor_mask = (region_dist < self.cutoff)

                    for i in range(len(neighbor_mask)):
                        if neighbor_mask[i]:
                            res = self.residue_atoms[i]

                            neighbor_mask[self.residue_atoms == res] = True

                    neighbor_mask[self.indices] = False

                    charges_cut = charges.unsqueeze(1).repeat(1, len(self.indices))
                    esp = torch.sum((charges_cut[neighbor_mask, :]/distance[neighbor_mask, :]), dim=0)

                    grad_mm[neighbor_mask[mask], :, :] = -1 * (charges_cut[neighbor_mask, :] * ((distance[neighbor_mask, :])**(-3))).unsqueeze(2) * r[neighbor_mask, :, :]

                    reduced_field = torch.sum(grad_mm, dim=0)
                    for i in range(len(self.indices)):
                        elec_field[i, i] = reduced_field[i]

                    grad_mm = grad_mm.permute(1, 0, 2)

                else:
                    esp = torch.ones_like(ml_charges, dtype=self.default_dtype)
                    grad_mm = torch.zeros((ml_charges.shape[0], mm_positions.shape[0], mm_positions.shape[1]), 
                        dtype=self.default_dtype).to(positions.device)
                    elec_field = torch.zeros((ml_charges.shape[0], ml_charges.shape[0], positions.shape[1]), dtype=self.default_dtype).to(positions.device)

                    for i in range(len(self.indices)):
                        # Gradient of ESP on ML atom i w.r.t. MM atom positions: dVi/dx = -qi * (ri-r)^3 * (x-xi) 
                        # positions unit: Angstrom
                        # charge unit: Amber's Unit (E=q1*q2/r where E is in kcal/mol)
                        potential, grad_mm_site, field = self.calc_esp(positions, i, charges, self.indices, box_vectors, self.cutoff, self.alpha, self.kmax)
                        esp[i] = potential
                        grad_mm[i] = grad_mm_site
                        elec_field[:, i, :] = -1 * field

                esp = esp.unsqueeze(1)
                # print('ESP')
                # print(esp)
                # print('elec_field')
                # print(elec_field)

                # prepare input dict 
                input_dict={}

                if boxvectors is not None:
                    input_dict["cell"]=boxvectors.to(dtype=self.default_dtype) * self.nm_to_distance
                    pbc = True
                else:
                    input_dict["cell"]=torch.eye(3, device=positions.device)
                    pbc = False

                input_dict["pbc"]=self.pbc
                # print(input_dict["pbc"])

                input_dict["atomic_numbers"] = self.atomic_numbers
                input_dict["atom_types"] = self.atom_types
                input_dict["pos"] = ml_positions
                input_dict["elec_potential"] = esp
                input_dict["grad_factor"] = elec_field
                # print(elec_field)

                # compute edges
                mapping, shifts_idx = simple_nl(ml_positions, input_dict["cell"], pbc, self.r_max)

                input_dict["edge_index"] = mapping
                input_dict["edge_cell_shift"] = shifts_idx

                out = self.model(input_dict)    

                # return energy and forces
                energy = (out["self_energy"]+out["coupl_energy"])*self.energy_to_kJ

                forces_padded[self.indices, :] = (out["self_forces"]+out["coupl_forces"]) * self.energy_to_kJ / self.distance_to_nm

                grad_esp = (out["Grad_ESP"]+out["Grad_ESP_coupl"])
                
                grad_esp_expand = grad_esp.unsqueeze(2).expand_as(grad_mm)
                
                forces_padded[mask, :] = torch.sum(grad_esp_expand * grad_mm, 0) * self.energy_to_kJ / self.distance_to_nm
                 
                return (energy, forces_padded)

        is_periodic = system.usesPeriodicBoundaryConditions()

        nequipforce = NepoIPdForce(self.model_path, includedAtoms, atoms, is_periodic, self.cut_esp, self.distance_to_nm, self.energy_to_kJ_per_mol, charges, alpha, cutoff, kmax, self.atom_types, **args)    
        
        # Convert it to TorchScript 
        module = torch.jit.script(nequipforce)

        # Create the TorchForce and add it to the System.
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)

        force.setOutputsForces(True)
        system.addForce(force)

MLPotential.registerImplFactory('nepoipd', NepoIPdPotentialImplFactory())
