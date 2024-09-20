from typing import NamedTuple
import jax.numpy as jnp
from jf1uids.geometry.geometry import CARTESIAN, SPHERICAL, CYLINDRICAL, _center_of_volume, _r_hat_alpha

# Helper data like the radii and cell volumes 
# in the simulation or cooling tables etc.

class HelperData(NamedTuple):
    geometric_centers: jnp.ndarray = None
    volumetric_centers: jnp.ndarray = None
    r_hat_alpha: jnp.ndarray = None
    cell_volumes: jnp.ndarray = None
    inner_cell_boundaries: jnp.ndarray = None
    outer_cell_boundaries: jnp.ndarray = None

def get_helper_data(config):
    dx = config.box_size / (config.num_cells - 1)
    if config.geometry == CARTESIAN:
        r = jnp.linspace(0, config.box_size, config.num_cells)
        r_hat = dx * jnp.ones_like(r) # not really
        cell_volumes = dx * jnp.ones_like(r)
        inner_cell_boundaries = r - dx / 2
        outer_cell_boundaries = r + dx / 2
        return HelperData(geometric_centers = r, r_hat_alpha = r_hat, cell_volumes = cell_volumes, inner_cell_boundaries = inner_cell_boundaries, outer_cell_boundaries = outer_cell_boundaries)
    elif config.geometry == SPHERICAL or config.geometry == CYLINDRICAL:
        # r = jnp.linspace(- 3 * dx/2, config.box_size - 3 * dx / 2, config.num_cells)
        r = jnp.linspace(dx / 2, config.box_size + dx / 2, config.num_cells)
        inner_cell_boundaries = r - dx / 2
        outer_cell_boundaries = r + dx / 2
        volumetric_centers = _center_of_volume(r, dx, config.geometry)
        r_hat = _r_hat_alpha(r, dx, config.geometry)
        cell_volumes = 2 * config.geometry * jnp.pi * dx * r_hat
        return HelperData(geometric_centers = r, volumetric_centers = volumetric_centers, r_hat_alpha = r_hat, cell_volumes = cell_volumes, inner_cell_boundaries = inner_cell_boundaries, outer_cell_boundaries = outer_cell_boundaries)