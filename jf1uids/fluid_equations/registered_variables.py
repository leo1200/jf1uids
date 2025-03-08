from typing import NamedTuple

import jax.numpy as jnp

from jaxtyping import Array, Float, Int

from typing import Union

from jf1uids.option_classes.simulation_config import SimulationConfig

class StaticIntVector(NamedTuple):
    x: int
    y: int
    z: int

class RegisteredVariables(NamedTuple):
    """The registered variables are the variables that are
    stored in the state array. The order of the variables
    in the state array is important and should be consistent
    throughout the code.
    """

    #: Number of variables
    num_vars: int = 3

    # Baseline variables
    
    #: Density index
    density_index: int = 0

    #: Velocity index
    velocity_index: Union[int, StaticIntVector] = 1
    # in e.g. 3D, we have three velocity components, each with its own index

    #: Magnetic field index
    magnetic_index: Union[int, StaticIntVector] = -1

    #: Energy index
    pressure_index: int = 2

    # Additional variables, these
    # have to be registered

    #: stellar wind density index
    wind_density_index: int = -1
    wind_density_active: bool = False

    #: simplified cosmic rays
    # in the simplest CR model witout CR diffusion,
    # streaming and no explicitly modeled magnetic field
    # n_CR = P_CR^(1/gamma_CR) is a conserved quantity.
    # This is the cosmic_ray_n, the index below points to.
    cosmic_ray_n_index: int = -1
    cosmic_ray_n_active: bool = False

    # here you can add more variables


def get_registered_variables(config: SimulationConfig) -> RegisteredVariables:
    """Get the registered variables for the simulation.

    Args:
        config: The simulation configuration.

    Returns:
        The registered variables.
    """

    registered_variables = RegisteredVariables()

    if config.dimensionality == 2:

        # we have two velocity components
        registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 1)

        # update the velocity index
        registered_variables = registered_variables._replace(velocity_index = StaticIntVector(1, 2, -1))

        # TODO: unified MHD approach in 1D/2D/3D
        # magnetic field index
        if config.mhd:
            # TODO: better indexing
            registered_variables = registered_variables._replace(pressure_index = 3)
            registered_variables = registered_variables._replace(magnetic_index = StaticIntVector(4, 5, 6))
            registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 3)
        else:
            # update the pressure index
            registered_variables = registered_variables._replace(pressure_index = registered_variables.num_vars - 1)

    if config.dimensionality == 3:
        
        # we have three velocity components
        registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 2)

        # update the velocity index to be an array
        registered_variables = registered_variables._replace(velocity_index = StaticIntVector(1, 2, 3))

        # update the pressure index
        registered_variables = registered_variables._replace(pressure_index = registered_variables.num_vars - 1)

    if config.wind_config.trace_wind_density:
        registered_variables = registered_variables._replace(wind_density_index = registered_variables.num_vars)
        registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 1)
        registered_variables = registered_variables._replace(wind_density_active = True)

    if config.cosmic_ray_config.cosmic_rays:
        registered_variables = registered_variables._replace(cosmic_ray_n_index = registered_variables.num_vars)
        registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 1)
        registered_variables = registered_variables._replace(cosmic_ray_n_active = True)

    # here you can register more variables

    return registered_variables