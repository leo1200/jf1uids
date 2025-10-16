from typing import Union
import jax
from functools import partial
import jax.numpy as jnp

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector import (
#     _cnn_mhd_corrector,
# )
# from corrector_src.model._cnn_mhd_corrector import (
#     _cnn_mhd_corrector_3d,
#     _cnn_mhd_corrector_2d,
# )
from jf1uids._physics_modules._cooling._cooling import update_pressure_by_cooling
from jf1uids._physics_modules._cosmic_rays.cr_injection import (
    inject_crs_at_strongest_shock,
)
from jf1uids._physics_modules._neural_net_force._neural_net_force import (
    _neural_net_force,
)
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids._physics_modules._stellar_wind.stellar_wind import _wind_injection
from jf1uids.shock_finder.shock_finder import shock_criteria

import equinox as eqx


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _run_physics_modules(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
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
        primitive_state = _wind_injection(
            primitive_state, dt, config, params, helper_data, registered_variables
        )

        # we might want to run the boundary handler after all physics modules have completed
        # primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    if config.cosmic_ray_config.diffusive_shock_acceleration:
        shock_crit = shock_criteria(
            primitive_state, config, registered_variables, helper_data
        )

        # injecting cosmic rays only after a certain amount of time
        # is an ad-hoc fix to problems that come about when a shock
        # has not yet properly formed
        primitive_state = jax.lax.cond(
            jnp.logical_and(
                current_time
                >= params.cosmic_ray_params.diffusive_shock_acceleration_start_time,
                jnp.any(shock_crit),
            ),
            lambda primitive_state: inject_crs_at_strongest_shock(
                primitive_state,
                params.gamma,
                helper_data,
                params.cosmic_ray_params,
                config,
                registered_variables,
                dt,
            ),
            lambda primitive_state: primitive_state,
            primitive_state,
        )

    if config.cooling_config.cooling:
        primitive_state = update_pressure_by_cooling(
            primitive_state,
            registered_variables,
            config.cooling_config.cooling_curve_config,
            params,
            dt,
        )

    if config.neural_net_force_config.neural_net_force:
        primitive_state = _neural_net_force(
            primitive_state,
            config,
            registered_variables,
            params,
            helper_data,
            dt,
            current_time,
        )

    # if config.cnn_mhd_corrector_config.cnn_mhd_corrector:
    #     if config.dimensionality == 2:
    #         primitive_state = _cnn_mhd_corrector_2d(
    #             primitive_state, config, registered_variables, params, dt
    #         )
    #     elif config.dimensionality == 3:
    #         primitive_state = _cnn_mhd_corrector_3d(
    #             primitive_state, config, registered_variables, params, dt
    #         )

    if config.corrector_config.corrector:
        neural_net_params = params.corrector_params.network_params
        neural_net_static = config.corrector_config.network_static
        model = eqx.combine(neural_net_params, neural_net_static)

        primitive_state = model(
            primitive_state, config, registered_variables, params, dt
        )
    """
    if config.runtime_debugging:
        checkify.check(
            jax.numpy.any(jax.numpy.isnan(primitive_state)),
            "nan found after mhd corrector after corrector",
        )
        checkify.check(
            jax.numpy.any(
                primitive_state[registered_variables.pressure_index, ...] < 0
            ),
            "negative value found in pressure after corrector",
        )
        checkify.check(
            jax.numpy.any(primitive_state[registered_variables.density_index, ...] < 0),
            "negative value found in density after corrector",
        )
    """
    return primitive_state
