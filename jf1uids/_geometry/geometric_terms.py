# general imports
from functools import partial
import jax
import jax.numpy as jnp

# typechecking
from typing import Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# general jf1uids imports
from jf1uids.option_classes.simulation_config import STATE_TYPE, STATE_TYPE_ALTERED, SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# limited gradients
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _pressure_nozzling_source(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> STATE_TYPE_ALTERED:
    """Pressure nozzling source term as of the geometry of the domain.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        helper_data: The helper data.
        registered_variables: The registered variables.

    Returns:
        The pressure nozzling source
    """
    
    # get the pressure
    p = primitive_state[registered_variables.pressure_index]

    # calculate the limited pressure gradients
    dp_dr = _calculate_limited_gradients(primitive_state, config, helper_data, axis = 1)[registered_variables.pressure_index]

    # calculate the pressure nozzling term, following
    # eq. 14 in Crittendend and Balachandar, 2018
    # https://doi.org/10.1007/s00193-017-0784-y
    pressure_nozzling = helper_data.geometric_centers[1:-1] ** (config.geometry - 1) * p[1:-1] + (helper_data.r_hat_alpha[1:-1] - helper_data.volumetric_centers[1:-1] * helper_data.geometric_centers[1:-1] ** (config.geometry - 1)) * dp_dr
    nozzling = jnp.zeros((registered_variables.num_vars, p.shape[0] - 2))
    nozzling = nozzling.at[registered_variables.velocity_index].set(config.geometry * pressure_nozzling)

    return nozzling