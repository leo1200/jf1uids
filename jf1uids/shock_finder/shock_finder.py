from functools import partial
import jax.numpy as jnp
import jax

# @partial(jax.jit, static_argnames=['registered_variables'])
# def shock_criteria1D(primitive_state, registered_variables):
#     """
#     Compute the shock criteria for each cell in the domain.
#     """

#     # Extract the pressure
#     velocity = primitive_state[registered_variables.velocity_index]
#     pressure = primitive_state[registered_variables.pressure_index]
#     density = primitive_state[registered_variables.density_index]

#     # central difference 1d divergence
#     div_v = jnp.zeros_like(velocity)
#     div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2)

#     return div_v < 0

@jax.jit
def shock_sensor(pressure):
    """
    WENO-JS smoothness indicator for shock detection.
    """
    shock_sensors = jnp.zeros_like(pressure)
    shock_sensors = shock_sensors.at[1:-1].set((1/4 * (pressure[2:] - pressure[:-2])**2 + 13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2])))
    return shock_sensors

@jax.jit
def find_shock_zone(pressure, velocity):

    num_cells = pressure.shape[0]

    sensors = shock_sensor(pressure)

    div_v = jnp.zeros_like(velocity)
    div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2)

    sensors = jnp.where(div_v < 0, sensors, 0)

    max_shock_idx = jnp.argmax(sensors)

    bound_val = 0.05 * jnp.max(sensors)

    to_next_pressure_differences = jnp.zeros_like(pressure)
    to_next_pressure_differences = jnp.abs(to_next_pressure_differences.at[:-1].set(pressure[1:] - pressure[:-1]))

    bound_diff = 0.1 * to_next_pressure_differences[max_shock_idx]

    # find the first index left and right
    # of the max_shock_idx where the sensor is
    # smaller than bound

    indices = jnp.arange(num_cells)
    left_indices = jnp.where((indices < max_shock_idx) & ((to_next_pressure_differences < bound_diff)), indices, -1)
    right_indices = jnp.where((indices > max_shock_idx) & ((sensors < bound_val) | (to_next_pressure_differences < bound_diff)), indices, num_cells)

    left_idx = jnp.max(left_indices)
    right_idx = jnp.min(right_indices)

    return left_idx, right_idx


# @partial(jax.jit, static_argnames=['registered_variables'])
# def strongest_shock_radius(primitive_state, helper_data, registered_variables):
#     """
#     Compute the radius of the strongest shock in the domain.
#     """
#     r = helper_data.geometric_centers
#     shock_sensors = shock_sensor(primitive_state[registered_variables.pressure_index])
#     max_shock_idx = jnp.argmax(shock_sensors)
#     return (r[max_shock_idx - 1] * shock_sensors[max_shock_idx - 1] + r[max_shock_idx] * shock_sensors[max_shock_idx] + r[max_shock_idx + 1] * shock_sensors[max_shock_idx + 1]) / (shock_sensors[max_shock_idx - 1] + shock_sensors[max_shock_idx] + shock_sensors[max_shock_idx + 1])

