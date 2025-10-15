import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# general
from functools import partial
import jax.numpy as jnp
import jax
import equinox as eqx

from jax.experimental import checkify

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

from jf1uids._physics_modules._mhd._vector_maths import curl2D, curl3D, divergence3D
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _cnn_mhd_corrector_2d(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    time_step: Float[Array, ""],
):
    # scale_factor = 1e-2
    neural_net_params = params.cnn_mhd_corrector_params.network_params
    neural_net_static = config.cnn_mhd_corrector_config.network_static
    model = eqx.combine(neural_net_params, neural_net_static)

    correction = model(primitive_state)

    # to not add divergence errors, we learn a correction for the electric field
    # - and the divergence of a curl is zero
    electric_field_correction = correction[-3:, ...]
    magnetic_field_correction = curl2D(electric_field_correction, config.grid_spacing)
    correction = correction.at[-3:, ...].set(magnetic_field_correction)

    # update the primitive state with the correction
    primitive_state = primitive_state + correction * time_step

    # ensure that the pressure is larger than a minimum value
    p_min = 1e-4
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
    )

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _cnn_mhd_corrector_3d(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    time_step: Float[Array, ""],
):
    # scale_factor = 1e-2
    neural_net_params = params.cnn_mhd_corrector_params.network_params
    neural_net_static = config.cnn_mhd_corrector_config.network_static
    model = eqx.combine(neural_net_params, neural_net_static)

    correction = model(primitive_state)
    # correction = jnp.tanh(correction) * scale_factor

    # to not add divergence errors, we learn a correction for the electric field
    # - and the divergence of a curl is zero
    electric_field_correction = correction[-3:, ...]
    magnetic_field_correction = curl3D(electric_field_correction, config.grid_spacing)
    correction = correction.at[-3:, ...].set(magnetic_field_correction)
    if config.runtime_debugging:
        jax.debug.print(
            "divergence of correction {}",
            jnp.max(divergence3D(correction[-3:, ...], config.grid_spacing)),
        )
    # update the primitive state with the correction
    primitive_state = primitive_state + correction * time_step

    # ensure that the pressure is larger than a minimum value
    p_min = 1e-12
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
    )
    rho_min = 1e-12
    primitive_state = primitive_state.at[registered_variables.density_index].set(
        jnp.maximum(primitive_state[registered_variables.density_index], rho_min)
    )

    return primitive_state
