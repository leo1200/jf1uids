from typing import NamedTuple
import jax.numpy as jnp
from jf1uids._geometry.geometry import CARTESIAN, SPHERICAL, CYLINDRICAL, _center_of_volume, _r_hat_alpha
from jf1uids.option_classes.simulation_config import SimulationConfig

# Helper data like the radii and cell volumes 
# in the simulation or cooling tables etc.

class HelperData(NamedTuple):
    """Helper data used throughout the simulation."""

    #: The geometric centers of the cells.
    geometric_centers: jnp.ndarray = None

    #: The volumetric centers of the cells.
    #: Same as the geometric centers for Cartesian geometry.
    volumetric_centers: jnp.ndarray = None

    #: cell center to box center distances
    #: only for config.dimensionality > 1
    r: jnp.ndarray = None

    #: A helper variable, defined as
    #: \hat{r}^\alpha = V_j / (2 * \alpha * \pi * \Delta r)
    #: with V_j the volume of cell j, \alpha the geometry factor
    #: and \Delta r the cell width.
    r_hat_alpha: jnp.ndarray = None

    #: The cell volumes.
    cell_volumes: jnp.ndarray = None

    #: Coordinates of the inner cell boundaries.
    inner_cell_boundaries: jnp.ndarray = None

    #: Coordinates of the outer cell boundaries.
    outer_cell_boundaries: jnp.ndarray = None

def get_helper_data(config: SimulationConfig) -> HelperData:
    """Generate the helper data for the simulation from the configuration."""

    if config.dimensionality > 1:
        x = jnp.linspace(0, config.box_size, config.num_cells)
        y = jnp.linspace(0, config.box_size, config.num_cells)
        z = jnp.linspace(0, config.box_size, config.num_cells)
        geometric_centers = jnp.meshgrid(x, y, z)

        # calculate the distances from the cell centers to the box center
        box_center = jnp.array([0.5, 0.5, 0.5]) * config.box_size
        geometric_centers = jnp.array(geometric_centers)
        geometric_centers = jnp.moveaxis(geometric_centers, 0, -1)
        volumetric_centers = geometric_centers

        r = jnp.linalg.norm(geometric_centers - box_center, axis = -1)

        return HelperData(geometric_centers = geometric_centers, volumetric_centers = volumetric_centers, r = r)

    dx = config.box_size / (config.num_cells - 1)
    if config.geometry == CARTESIAN:
        r = jnp.linspace(0, config.box_size, config.num_cells)
        r_hat = dx * jnp.ones_like(r) # not really
        cell_volumes = dx * jnp.ones_like(r)
        inner_cell_boundaries = r - dx / 2
        outer_cell_boundaries = r + dx / 2
        return HelperData(geometric_centers = r, r_hat_alpha = r_hat, cell_volumes = cell_volumes, inner_cell_boundaries = inner_cell_boundaries, outer_cell_boundaries = outer_cell_boundaries, volumetric_centers = r)
    elif config.geometry == SPHERICAL or config.geometry == CYLINDRICAL:
        # r = jnp.linspace(- 3 * dx/2, config.box_size - 3 * dx / 2, config.num_cells)
        r = jnp.linspace(dx / 2, config.box_size + dx / 2, config.num_cells)
        inner_cell_boundaries = r - dx / 2
        outer_cell_boundaries = r + dx / 2
        volumetric_centers = _center_of_volume(r, dx, config.geometry)
        r_hat = _r_hat_alpha(r, dx, config.geometry)
        cell_volumes = 2 * config.geometry * jnp.pi * dx * r_hat
        return HelperData(geometric_centers = r, volumetric_centers = volumetric_centers, r_hat_alpha = r_hat, cell_volumes = cell_volumes, inner_cell_boundaries = inner_cell_boundaries, outer_cell_boundaries = outer_cell_boundaries)