from typing import NamedTuple

import jax.numpy as jnp

from jf1uids.option_classes.simulation_config import SimulationConfig

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

    #: pressure index
    pressure_index: int = 1

    #: Energy index
    energy_index: int = 2

    # Additional variables, these
    # have to be registered

    #: stellar wind density index
    wind_density_index: int = jnp.inf

    # here you can add more variables


def get_registered_variables(config: SimulationConfig) -> RegisteredVariables:
    """Get the registered variables for the simulation.

    Args:
        config: The simulation configuration.

    Returns:
        The registered variables.
    """

    registered_variables = RegisteredVariables()

    if config.wind_config.trace_wind_density:
        registered_variables = registered_variables._replace(wind_density_index = registered_variables.num_vars)
        registered_variables = registered_variables._replace(num_vars = registered_variables.num_vars + 1)

    # here you can register more variables

    return registered_variables