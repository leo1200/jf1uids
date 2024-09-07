from typing import NamedTuple
import jax.numpy as jnp
from jf1uids.geometry import center_of_volume, r_hat_alpha

# Helper data like the radii and cell volumes 
# in the simulation or cooling tables etc.

class SimulationHelperData(NamedTuple):
    geometric_centers: jnp.ndarray = None
    volumetric_centers: jnp.ndarray = None
    r_hat_alpha: jnp.ndarray = None

def get_helper_data(config):
    dx = config.box_size / config.num_cells
    if config.alpha_geom == 0:
        r = jnp.linspace(0, config.box_size, config.num_cells)
        r_hat_alpha = dx * jnp.ones_like(r) # not really
        return SimulationHelperData(geometric_centers = r, r_hat_alpha = r_hat_alpha)
    if config.alpha_geom == 2:
        r = jnp.linspace(- 3 * dx/2, config.box_size - 3 * dx / 2, config.num_cells)
        volumetric_centers = center_of_volume(r, dx, config.alpha_geom)
        r_hat = r_hat_alpha(r, dx, config.alpha_geom)
        return SimulationHelperData(geometric_centers = r, volumetric_centers = volumetric_centers, r_hat_alpha = r_hat)