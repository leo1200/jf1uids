import numpy as np

from numba import njit
from numba import prange

import jax.numpy as jnp

from jf1uids.initial_condition_generation.turb import TurbGen

@njit(parallel=True)
def turbulence_creator(cell_centers, slope, kmin, kmax, spect_form = 2, angles_exp = 1, sol_weight = 0.5, seeed = None):
    turb = TurbGen()
    L = [1.0, 1.0, 1.0]  # Box size
    ndim = 3.0

    turb.init_single_realisation_with_kmid(ndim, L, kmin, kmax, kmax, spect_form, slope, slope, angles_exp, sol_weight, seeed)

    # initialize velocity array of the same shape as cell_centers
    vfield = np.zeros(cell_centers.shape, dtype=np.float64)

    # calculate the velocity field at the cell centers
    for i in prange(cell_centers.shape[0]):
        vfield[i] = turb.get_turb_vector(cell_centers[i])

    return vfield

def create_turbulence(cell_centers, slope, kmin, kmax, spect_form = 2, angles_exp = 1, sol_weight = 0.5, seeed = None):
    
    # cast the jax array to numpy array
    cell_centers = np.asarray(cell_centers)
    
    cell_center_shape = cell_centers.shape
    
    # cell centers will be of the form 3 x num_cells x num_cells x num_cells
    # reshape to (num_cells**3, 3)
    cell_centers = cell_centers.reshape(-1, 3)

    # calculate the velocity field at the cell centers
    vfield = turbulence_creator(cell_centers, slope, kmin, kmax, spect_form, angles_exp, sol_weight, seeed)

    # reshape the velocity field to the original shape
    vfield = vfield.reshape(cell_center_shape)

    # back to jax array
    vfield = jnp.asarray(vfield)

    return vfield