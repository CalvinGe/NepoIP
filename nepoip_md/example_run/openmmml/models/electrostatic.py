import torch
from typing import Tuple, List, Optional

def convert_optional_tensor(optional_tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if optional_tensor is None:
        # Handle the None case, e.g., return a zero tensor with a specific shape
        return torch.tensor([])
    else:
        return optional_tensor

@torch.jit.script
def compute_batch(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor, cell: torch.Tensor, r: torch.Tensor, charges: torch.Tensor, alpha: float, volume: float) -> torch.Tensor:
    k_vecs = torch.stack((kx, ky, kz), dim=1)
    k_vecs = k_vecs / torch.diagonal(cell)
    k2 = torch.sum(k_vecs ** 2, dim=1)
    img = 2 * torch.pi * torch.matmul(r, k_vecs.t())
    real = - (torch.pi / alpha) ** 2 * k2

    terms = torch.sum(charges[:, None] * torch.exp(real) * torch.cos(img), dim=0) / k2 / volume / torch.pi
    return terms

@torch.jit.script
def parallel_compute(kmax: torch.Tensor, cell: torch.Tensor, r: torch.Tensor, charges: torch.Tensor, alpha: float, volume: float, batch_size: int) -> torch.Tensor:
    # Generate all combinations of kx, ky, kz
    k_range_x = torch.arange(-kmax[0], kmax[0] + 1)
    k_range_y = torch.arange(-kmax[1], kmax[1] + 1)
    k_range_z = torch.arange(-kmax[2], kmax[2] + 1)
    kx, ky, kz = torch.meshgrid(k_range_x, k_range_y, k_range_z, indexing='ij')
    kx, ky, kz = kx.flatten(), ky.flatten(), kz.flatten()
    
    # Filter out the zero vector
    non_zero_mask = (kx != 0) | (ky != 0) | (kz != 0)
    kx, ky, kz = kx[non_zero_mask], ky[non_zero_mask], kz[non_zero_mask]

    rec_sum = torch.tensor(0.0)

    # Process in batches
    num_elements = kx.size(0)
    # print(num_elements)
    for i in range(0, num_elements, batch_size):
        end = min(i + batch_size, num_elements)
        kx_batch = kx[i:end]
        ky_batch = ky[i:end]
        kz_batch = kz[i:end]
        terms = compute_batch(kx_batch, ky_batch, kz_batch, cell, r, charges, alpha, volume)
        rec_sum += torch.sum(terms)
    
    return rec_sum

@torch.jit.script
def ewald_esp(positions: torch.Tensor, index: int, charges: torch.Tensor, ml_indices: torch.Tensor, cell: torch.Tensor, cutoff: float, alpha: float, kmax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    target_ind = ml_indices[index]
    # print(target_ind)

    self_mask = torch.zeros(charges.shape, dtype=torch.bool)
    self_mask[target_ind] = True

    ml_mask = torch.zeros(charges.shape, dtype=torch.bool)
    ml_mask[ml_indices] = True

    ml_self_mask = torch.zeros(ml_indices.shape, dtype=torch.bool)
    ml_self_mask[index] = True

    new_indices = ml_indices[~ml_self_mask]
    new_indices[new_indices>target_ind] -= 1
    

    point = positions[target_ind]
    r = positions[~self_mask] - point

        # Periodic Boundary Condition. Requires cutoff < box_length/2
    for j in range(3):
        box_length = cell[j][j]
        r[:, j] -= round(r[:, j] / box_length) * box_length 

    # # Calculate the direct space sum and its derivative   
    
    charges_cut = charges.clone()
    charges_cut = charges_cut[~self_mask]
    
    distance = torch.sum(r ** 2, 1) ** 0.5
    neighbor_mask = (distance < cutoff)

    # charges_cut[~neighbor_mask] = 0 # Set zero charges for atoms out of cutoff

    erfc_term = torch.special.erfc(alpha * distance[neighbor_mask]) 
    potential = torch.sum(charges_cut[neighbor_mask] / distance[neighbor_mask] * erfc_term)

    # Calculate the reciprocal space sum and its derivative  
    volume = cell[0][0] * cell[1][1] * cell[2][2] 

    rec_sum = parallel_compute(kmax, cell, positions - point, charges, alpha, volume, 256)

    potential += rec_sum

    # self energy
    self_interaction = alpha / torch.sqrt(torch.pi) * charges[self_mask]
    potential -= 2 * self_interaction.squeeze()

    # Exclude the intra-molecular coulomb interaction
    intra_potential = torch.sum(charges_cut[new_indices]/distance[new_indices]) 
    potential -= intra_potential 

    # Gradients
    grad_auto = torch.autograd.grad([potential], [positions], retain_graph=True)[0]
    grad_auto = convert_optional_tensor(grad_auto)

    grad_mm_site = grad_auto[~ml_mask]

    elec_field = grad_auto[ml_mask]

    return potential, grad_mm_site, elec_field

@torch.jit.script
def nopbc_esp(positions: torch.Tensor, index: int, charges: torch.Tensor, ml_indices: torch.Tensor, cell: Optional[torch.Tensor], cutoff: float, alpha: float, kmax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    target_ind = ml_indices[index]
    # print(target_ind)

    self_mask = torch.zeros(charges.shape, dtype=torch.bool)
    self_mask[target_ind] = True

    ml_mask = torch.zeros(charges.shape, dtype=torch.bool)
    ml_mask[ml_indices] = True

    ml_self_mask = torch.zeros(ml_indices.shape, dtype=torch.bool)
    ml_self_mask[index] = True

    new_indices = ml_indices[~ml_self_mask]
    new_indices[new_indices>target_ind] -= 1
    

    point = positions[target_ind]
    r = positions[~ml_mask] - point

    # # Calculate the direct space sum and its derivative   
    
    charges_cut = charges.clone()
    charges_cut = charges_cut[~ml_mask]
    
    distance = torch.sum(r ** 2, 1) ** 0.5

    potential = torch.sum(charges_cut / distance)

    # Gradients
    grad_auto = torch.autograd.grad([potential], [positions], retain_graph=True)[0]
    # grad_auto = torch.autograd.grad([intra_potential], [positions], retain_graph=True)[0]
    grad_auto = convert_optional_tensor(grad_auto)

    grad_mm_site = grad_auto[~ml_mask]

    elec_field = grad_auto[ml_mask]

    return potential, grad_mm_site, elec_field
