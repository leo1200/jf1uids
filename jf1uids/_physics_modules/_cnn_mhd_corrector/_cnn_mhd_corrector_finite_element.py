import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# general
from functools import partial
import jax.numpy as jnp
import jax

import equinox as eqx

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

from jf1uids._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
    update_cell_center_fields,
)
from jf1uids._finite_difference._maths._differencing import finite_difference_int6
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams


def finite_difference_curl_3D(omega_bar):
    dtdy = 1.0
    dtdz = 1.0
    dtdx = 1.0
    rhs_bx = -dtdy * finite_difference_int6(
        omega_bar[2], YAXIS
    ) + dtdz * finite_difference_int6(omega_bar[1], ZAXIS)

    rhs_by = -dtdz * finite_difference_int6(
        omega_bar[0], ZAXIS
    ) + dtdx * finite_difference_int6(omega_bar[2], XAXIS)

    rhs_bz = -dtdx * finite_difference_int6(
        omega_bar[1], XAXIS
    ) + dtdy * finite_difference_int6(omega_bar[0], YAXIS)
    return rhs_bx, rhs_by, rhs_bz


class CorrectorCNN(eqx.Module):
    """
    A simple CNN that maps an input of shape (C, H, W) to an output of the same shape.
    """

    layers: list

    def __init__(self, in_channels: int, hidden_channels: int, *, key: PRNGKeyArray):
        # We need a key for each convolutional layer
        key1, key2, key3 = jax.random.split(key, 3)

        # A simple 3-layer CNN.
        self.layers = (
            eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv3d(
                hidden_channels, in_channels - 3, 3, padding=1, key=key3, use_bias=False
            ),
        )

    def __call__(self,     
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ) -> Float[Array, "num_vars h w"]:
        """
        The forward pass of the model.
        """
        # Pass the input through the network to get the correction term
        neural_net_params = params.cnn_mhd_corrector_params.network_params
        neural_net_static = config.cnn_mhd_corrector_config.network_static
        model = eqx.combine(neural_net_params, neural_net_static)

        correction = model(primitive_state)

        omega_bar = correction[-3:, ...]
        bx_interface_correction, by_interface_correction, bz_interface_correction = (
            finite_difference_curl_3D(omega_bar)
        )
        correction = correction.at[-3:, ...].set(bx_interface_correction)
        correction = correction.at[-2:, ...].set(by_interface_correction)
        correction = correction.at[-1:, ...].set(bz_interface_correction)

        # update the primitive state with the correction
        primitive_state = primitive_state.at[:-6] + correction.at[:-3] * time_step
        primitive_state = primitive_state.at[:-3] + correction.at[-3:] * time_step

        primitive_state = update_cell_center_fields(
            primitive_state,
            primitive_state.at[-3],
            primitive_state.at[-2],
            primitive_state.at[-1],
            registered_variables,
        )

        p_min = 1e-20
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
        )

        return primitive_state