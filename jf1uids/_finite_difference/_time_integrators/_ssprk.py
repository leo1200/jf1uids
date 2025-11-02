from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jf1uids._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)
from jf1uids.variable_registry.registered_variables import RegisteredVariables


@partial(jax.jit, static_argnames=["registered_variables"])
def _ssprk4(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    def L(state):
        F_x = _weno_flux_x(state, gamma, registered_variables)
        F_y = _weno_flux_y(state, gamma, registered_variables)
        F_z = _weno_flux_z(state, gamma, registered_variables)
        return (
            -(F_x - jnp.roll(F_x, 1, axis=1)) / grid_spacing
            - (F_y - jnp.roll(F_y, 1, axis=2)) / grid_spacing
            - (F_z - jnp.roll(F_z, 1, axis=3)) / grid_spacing
        )

    # Coefficients: Spiteri & Ruuth (2002), optimal SSPRK(5,4)
    a1 = 0.391752226571890
    a2 = 0.482573533092014
    a3 = 0.435866521508459
    a4 = 0.282706005359643
    a5 = 0.066518647293668
    b1 = 0.368410593050371
    b2 = 0.251891774271694
    b3 = 0.544974750212370

    q0 = conserved_state

    q1 = q0 + a1 * dt * L(q0)
    q2 = b1 * q0 + (1 - b1) * q1 + a2 * dt * L(q1)
    q3 = b2 * q1 + (1 - b2) * q2 + a3 * dt * L(q2)
    q4 = b3 * q2 + (1 - b3) * q3 + a4 * dt * L(q3)
    q5 = a5 * q4 + (1 - a5) * q3 + a5 * dt * L(q4)

    return q5