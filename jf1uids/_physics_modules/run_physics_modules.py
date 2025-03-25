from typing import Union
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jf1uids._physics_modules._cosmic_rays.cr_injection import inject_crs_at_strongest_shock
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SPHERICAL, STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids._physics_modules._stellar_wind.stellar_wind import _wind_injection

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _run_physics_modules(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]]
) -> STATE_TYPE:
    """Run all the physics modules. The physics modules are switched on/off and
    configured in the simulation configuration. Parameters for the physics modules
    (with respect to which the simulation can be differentiated) are stored in the
    simulation parameters.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the physics modules applied.
    """
    
    # stellar wind
    if config.wind_config.stellar_wind:
        primitive_state = _wind_injection(primitive_state, dt, config, params, helper_data, registered_variables)

        # we might want to run the boundary handler after all physics modules have completed
        # primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    if config.cosmic_ray_config.diffusive_shock_acceleration:

        # injecting cosmic rays only after a certain amount of time
        # is an ad-hoc fix to problems that come about when a shock
        # has not yet properly formed
        primitive_state = jax.lax.cond(
            current_time > params.cosmic_ray_params.diffusive_shock_acceleration_start_time,
            lambda primitive_state: inject_crs_at_strongest_shock(primitive_state, params.gamma, helper_data, params.cosmic_ray_params, config, registered_variables, dt),
            lambda primitive_state: primitive_state,
            primitive_state
        )
    
    return primitive_state
