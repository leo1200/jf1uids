import jax.numpy as jnp

def shock_sensor(primitive_state):
    """
    WENO-JS smoothness indicator for shock detection.
    """
    _, _, p = primitive_state
    return (1/4 * (p[2:] - p[:-2])**2 + 13 / 12 * (p[2:] - 2 * p[1:-1] + p[:-2]))

def strongest_shock_radius(primitive_state, helper_data, padL, padR):
    """
    Compute the radius of the strongest shock in the domain.
    """
    r = helper_data.geometric_centers[padL + 1: -padR - 1]
    shock_sensors = shock_sensor(primitive_state)[padL: -padR]
    max_shock_idx = jnp.argmax(shock_sensors)
    return (r[max_shock_idx - 1] * shock_sensors[max_shock_idx - 1] + r[max_shock_idx] * shock_sensors[max_shock_idx] + r[max_shock_idx + 1] * shock_sensors[max_shock_idx + 1]) / (shock_sensors[max_shock_idx - 1] + shock_sensors[max_shock_idx] + shock_sensors[max_shock_idx + 1])

def strongest_shock_dissipated_energy_flux(primitive_state, gamma, padL, padR, statePad):
    """
    Compute the dissipated energy flux due to the strongest shock in the domain.
    """
    shock_sensors = shock_sensor(primitive_state)[padL: -padR]
    primitive_state = primitive_state[:, padL + 1: -padR - 1]
    max_shock_idx = jnp.argmax(shock_sensors)
    
    # get the pre and post shock states
    rho1, v1, P1 = primitive_state[:, max_shock_idx + statePad]
    rho2, v2, P2 = primitive_state[:, max_shock_idx - statePad]

    c1 = jnp.sqrt(gamma * P1 / rho1)

    e_th1 = P1 / ((gamma - 1))
    e_th2 = P2 / ((gamma - 1))

    x_s = rho2 / rho1

    e_diss = e_th2 - e_th1 * x_s ** gamma

    M_1_sq = (P2 / P1 - 1) * x_s / (gamma * (x_s - 1))

    f_diss = e_diss * jnp.sqrt(M_1_sq) * c1 / x_s

    return f_diss
