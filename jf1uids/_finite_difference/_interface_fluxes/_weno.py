from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jf1uids._finite_difference._fluid_equations._eigen import _eigen_x
from jf1uids._finite_difference._fluid_equations._fluxes import _mhd_flux_x
from jf1uids.variable_registry.registered_variables import RegisteredVariables

@jax.jit
def _weno_interpolate(
    a1, a2, a3, a4
):
    IS0 = 13 * (a1 - a2) ** 2 + 3 * (a1 - 3 * a2) ** 2
    IS1 = 13 * (a2 - a3) ** 2 + 3 * (a2 + a3) ** 2
    IS2 = 13 * (a3 - a4) ** 2 + 3 * (3 * a3 - a4) ** 2

    epsilon = 1e-8

    C0 = 0.1
    C1 = 0.6
    C2 = 0.3

    # also square Cs?
    alpha0 = (C0 / (epsilon + IS0)) ** 2
    alpha1 = (C1 / (epsilon + IS1)) ** 2
    alpha2 = (C2 / (epsilon + IS2)) ** 2

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return 1/3 * w0 * (a1 - 2 * a2 + a3) + 1/6 * (w2 - 1/2) * (a2 - 2 * a3 + a4)


@partial(jax.jit, static_argnames=["registered_variables"])
def _weno_flux_x(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    
    q = conserved_state

    lambdas, R, L = _eigen_x(q, gamma, registered_variables)

    F = _mhd_flux_x(q, gamma, registered_variables)

    # lambdas is of shape
    # (7, 32, 32, 32)

    # L is of shape
    # (7, 8, 32, 32, 32)

    # F is of shape
    # (8, 32, 32, 32)

    # F_char is of shape
    # (7, 32, 32, 32)

    F_char = jnp.einsum('abxyz,bxyz->axyz', L, F)
    q_char = jnp.einsum('abxyz,bxyz->axyz', L, q)

    deltaF_char = jnp.roll(F_char, -1, axis=1) - F_char
    deltaq_char = jnp.roll(q_char, -1, axis=1) - q_char

    # find the maximum of lambdas
    # along the x-axis in the stencil
    # i - 2 <= m <= i + 3 (sliding window)
    # -> shape (7, 32, 32, 32)
    
    # Stack shifted versions of lambdas for positions 
    # i-2, i-1, i, i+1, i+2, i+3
    # and take maximum along the stacked dimension
    lambda_stencil = jnp.stack([
        jnp.roll(lambdas, 2, axis=1),   # i-2
        jnp.roll(lambdas, 1, axis=1),   # i-1
        lambdas,                         # i
        jnp.roll(lambdas, -1, axis=1),  # i+1
        jnp.roll(lambdas, -2, axis=1),  # i+2
        jnp.roll(lambdas, -3, axis=1),  # i+3
    ], axis=0)
    
    # Take max over the stencil dimension (axis=0)
    lambda_max = jnp.max(lambda_stencil, axis=0)

    # Lax-Friedrichs flux splitting
    F_plus = 0.5 * (deltaF_char + lambda_max * deltaq_char) # F^+_{i+1/2}
    F_minus = 0.5 * (deltaF_char - lambda_max * deltaq_char) # F^-_{i+1/2}

    # WENO reconstruction for F_plus
    phi_plus = _weno_interpolate(
        jnp.roll(F_plus, 2, axis=1),
        jnp.roll(F_plus, 1, axis=1),
        F_plus,
        jnp.roll(F_plus, -1, axis=1),
    )

    # WENO reconstruction for F_minus
    phi_minus = _weno_interpolate(
        jnp.roll(F_minus, -2, axis=1),
        jnp.roll(F_minus, -1, axis=1),
        F_minus,
        jnp.roll(F_minus, 1, axis=1),
    )

    F_interface_char = phi_minus - phi_plus
    F_interface = jnp.einsum('abxyz,bxyz->axyz', R, F_interface_char)

    F_interface = 1/12 * (-jnp.roll(F, 1, axis=1) + 7 * F + 7 * jnp.roll(F, -1, axis=1) - jnp.roll(F, -2, axis=1)) + F_interface

    return F_interface

@partial(jax.jit, static_argnames=["registered_variables"])
def _weno_flux_y(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    # conserved_state is of shape (num_variables, nx, ny, nz)
    # need to permute to bring y-axis to x-axis position
    # and bring v_y to v_x position and B_y to B_x position
    qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]
    B_x = qy[registered_variables.magnetic_index.x]
    B_y = qy[registered_variables.magnetic_index.y]
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
    qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
    qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

    Fy = _weno_flux_x(qy, gamma, registered_variables)

    # need to permute back
    Fy = jnp.transpose(Fy, (0, 2, 1, 3))
    Fmomentum_x = Fy[registered_variables.momentum_index.x]
    Fmomentum_y = Fy[registered_variables.momentum_index.y]
    FB_x = Fy[registered_variables.magnetic_index.x]
    FB_y = Fy[registered_variables.magnetic_index.y]
    Fy = Fy.at[registered_variables.momentum_index.x].set(Fmomentum_y)
    Fy = Fy.at[registered_variables.momentum_index.y].set(Fmomentum_x)
    Fy = Fy.at[registered_variables.magnetic_index.x].set(FB_y)
    Fy = Fy.at[registered_variables.magnetic_index.y].set(FB_x)
    
    return Fy

@partial(jax.jit, static_argnames=["registered_variables"])
def _weno_flux_z(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    # conserved_state is of shape (num_variables, nx, ny, nz)
    # need to permute to bring z-axis to x-axis position
    # and bring v_z to v_x position and B_z to B_x position
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]
    B_x = qz[registered_variables.magnetic_index.x]
    B_z = qz[registered_variables.magnetic_index.z]
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
    qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
    qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

    Fz = _weno_flux_x(qz, gamma, registered_variables)

    # need to permute back
    Fz = jnp.transpose(Fz, (0, 3, 2, 1))
    Fmomentum_x = Fz[registered_variables.momentum_index.x]
    Fmomentum_z = Fz[registered_variables.momentum_index.z]
    FB_x = Fz[registered_variables.magnetic_index.x]
    FB_z = Fz[registered_variables.magnetic_index.z]
    Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
    Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)
    Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
    Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
    return Fz