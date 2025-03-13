# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from typing import Tuple, Union
from jf1uids.option_classes.simulation_config import FIELD_TYPE
from jaxtyping import Array, Int, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# NOTE: currently only works for 1d setups, TODO: generalize

@jax.jit
def shock_sensor(pressure: FIELD_TYPE) -> FIELD_TYPE:
    """
    WENO-JS 1D smoothness indicator for shock detection.

    Args:
        pressure: the 1d pressure

    Returns:
        shock sensors, high where large pressure jumps

    """

    shock_sensors = jnp.zeros_like(pressure)
    shock_sensors = shock_sensors.at[1:-1].set(
        1/4 * (pressure[2:] - pressure[:-2])**2 +
        13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2])
    )

    return shock_sensors

@jax.jit
def find_shock_zone(
    pressure: FIELD_TYPE,
    velocity: FIELD_TYPE
) -> Tuple[Union[int, Int[Array, ""]], Union[int, Int[Array, ""]], Union[int, Int[Array, ""]]]:
    """
    Find a numerically broadened shock region based of the strongest shock based
    on the result of the shock_sensor function and the pressure difference
    between adjacent cells. Assumes a shock front moving left to right.

    Args:
        pressure: 1d pressure
        velocity: 1d velocity

    Returns:
        index of max shock sensor,
        left boundary of broadened shock,
        right boundary of broadened shock
    
    """

    num_cells = pressure.shape[0]

    sensors = shock_sensor(pressure)

    # rule out regions with div v >= 0, see criterion 1 in Pfrommer et al, 2017
    div_v = jnp.zeros_like(velocity)
    div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2)
    sensors = jnp.where(div_v < 0, sensors, 0)

    max_shock_idx = jnp.argmax(sensors)

    # bound on the shock sensor
    bound_val = 0.05 * jnp.max(sensors)

    # calculate differences in pressure
    to_next_pressure_differences = jnp.zeros_like(pressure)
    to_next_pressure_differences = jnp.abs(to_next_pressure_differences.at[:-1].set(pressure[1:] - pressure[:-1]))

    # bound on the change in pressure between adjacent cells compared
    # to the pressure jump at the max_shock_index
    bound_diff = 0.1 * to_next_pressure_differences[max_shock_idx]

    # left index: closest left index where pressure_difference < bound_diff
    # right index: closest right index where sensor < bound_val or pressure_difference < bound_diff
    indices = jnp.arange(num_cells)
    left_indices = jnp.where((indices < max_shock_idx) & ((to_next_pressure_differences < bound_diff)), indices, -1)
    right_indices = jnp.where((indices > max_shock_idx) & ((sensors < bound_val) | (to_next_pressure_differences < bound_diff)), indices, num_cells)
    left_idx = jnp.max(left_indices)
    right_idx = jnp.min(right_indices)

    return max_shock_idx, left_idx, right_idx