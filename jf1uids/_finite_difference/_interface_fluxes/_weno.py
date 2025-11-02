from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jf1uids._finite_difference._fluid_equations._eigen import _eigen_x
from jf1uids._finite_difference._fluid_equations._fluxes import _mhd_flux_x
from jf1uids.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["registered_variables"])
def _weno_flux_x(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction matching Fortran.
    
    Returns the RHS contribution: -dtdl * (dF(i) - dF(i-1))
    But dtdl is applied in the time integrator, so this returns
    just the spatial derivative term to be used with the RK coefficients.
    
    Actually, looking at Fortran more carefully:
    - weno() computes qone(m,i) = -dtdl*(dF(m,i)-dF(m,i-1))
    - ssprk() then does: q(iw) = k1*q(iw) + k3*q(iw-1) + qone
    
    So weno() returns the FULL RHS including -dtdl factor.
    But the dtdl is computed in weno as: dtdl (passed as common variable).
    
    Actually wait - let me read ssprk more carefully:
    
    In ssprk:
      dtdx = k2*dt/dx
      dtdl = dtdx  (for x-sweep)
      call weno(qone,uone,a,F,R,L,bsy,bsz)
    
    In weno:
      qone(1,i) = -dtdl*(dF(1,i)-dF(1,i-1))
    
    Then back in ssprk:
      q(iw,1,ix,iy,iz) = k1*q(iw,1,ix,iy,iz)
         +                   +k3*q(iw-1,1,ix,iy,iz)+qone(1,ix)
    
    So yes, weno returns the FULL update including the -dtdl factor.
    But we're handling dtdl in the integrator, so let's return just the flux.
    """
    
    q = conserved_state
    N_vars, Nx, Ny, Nz = q.shape
    epsilon = 1e-8
    
    # Get eigenstructure
    lambdas_center, R, L = _eigen_x(q, gamma, registered_variables)
    
    # Get physical fluxes at cell centers
    F = _mhd_flux_x(q, gamma, registered_variables)
    
    # Fortran loops over interfaces i=0 to nn
    # For interface i between cells i and i+1, uses stencil: i-2, i-1, i, i+1, i+2, i+3
    # 
    # In our periodic domain with Nx cells, interface i is at the same position i
    # Cell i uses dF(i) - dF(i-1), so:
    #   - dF(0) - dF(Nx-1) for cell 0
    #   - dF(1) - dF(0) for cell 1
    #   etc.
    
    # For interface at position i, we need:
    # Cells: i-2, i-1, i, i+1, i+2, i+3 (wrapping with periodic BC)
    
    # Build stencils using rolls
    # For the array at position i to access i-2, we roll by +2
    F_stencil = jnp.stack([
        jnp.roll(F, 2, axis=1),   # i-2
        jnp.roll(F, 1, axis=1),   # i-1
        F,                         # i
        jnp.roll(F, -1, axis=1),  # i+1
        jnp.roll(F, -2, axis=1),  # i+2
        jnp.roll(F, -3, axis=1),  # i+3
    ], axis=0)  # Shape: (6, N_vars, Nx, Ny, Nz)
    
    q_stencil = jnp.stack([
        jnp.roll(q, 2, axis=1),
        jnp.roll(q, 1, axis=1),
        q,
        jnp.roll(q, -1, axis=1),
        jnp.roll(q, -2, axis=1),
        jnp.roll(q, -3, axis=1),
    ], axis=0)
    
    # Get maximum eigenvalue over stencil
    lambda_stencil = jnp.stack([
        jnp.roll(lambdas_center, 2, axis=1),
        jnp.roll(lambdas_center, 1, axis=1),
        lambdas_center,
        jnp.roll(lambdas_center, -1, axis=1),
        jnp.roll(lambdas_center, -2, axis=1),
        jnp.roll(lambdas_center, -3, axis=1),
    ], axis=0)
    
    amx = jnp.max(jnp.abs(lambda_stencil), axis=0)  # Shape: (7, Nx, Ny, Nz)
    
    # Transform stencil to characteristic variables
    # L has shape (7, N_vars, Nx, Ny, Nz) at interfaces
    # F_stencil: (6, N_vars, Nx, Ny, Nz)
    # Result: (6, 7, Nx, Ny, Nz)
    Fsk = jnp.einsum('mnxyz,snxyz->smxyz', L, F_stencil)
    qsk = jnp.einsum('mnxyz,snxyz->smxyz', L, q_stencil)
    
    # Compute differences
    dFsk = Fsk[1:, ...] - Fsk[:-1, ...]  # (5, 7, Nx, Ny, Nz)
    dqsk = qsk[1:, ...] - qsk[:-1, ...]
    
    # Base 4th-order flux
    first = (-Fsk[1, ...] + 7.0*Fsk[2, ...] + 7.0*Fsk[3, ...] - Fsk[4, ...]) / 12.0
    
    # Forward WENO (positive flux)
    aterm_p = 0.5 * (dFsk[0, ...] + amx * dqsk[0, ...])
    bterm_p = 0.5 * (dFsk[1, ...] + amx * dqsk[1, ...])
    cterm_p = 0.5 * (dFsk[2, ...] + amx * dqsk[2, ...])
    dterm_p = 0.5 * (dFsk[3, ...] + amx * dqsk[3, ...])
    
    IS0_p = 13.0 * (aterm_p - bterm_p)**2 + 3.0 * (aterm_p - 3.0*bterm_p)**2
    IS1_p = 13.0 * (bterm_p - cterm_p)**2 + 3.0 * (bterm_p + cterm_p)**2
    IS2_p = 13.0 * (cterm_p - dterm_p)**2 + 3.0 * (3.0*cterm_p - dterm_p)**2
    
    alpha0_p = 1.0 / (epsilon + IS0_p)**2
    alpha1_p = 6.0 / (epsilon + IS1_p)**2
    alpha2_p = 3.0 / (epsilon + IS2_p)**2
    
    alpha_sum_p = alpha0_p + alpha1_p + alpha2_p
    omega0_p = alpha0_p / alpha_sum_p
    omega2_p = alpha2_p / alpha_sum_p
    
    second = (omega0_p * (aterm_p - 2.0*bterm_p + cterm_p) / 3.0 
              + (omega2_p - 0.5) * (bterm_p - 2.0*cterm_p + dterm_p) / 6.0)
    
    # Backward WENO (negative flux)
    aterm_m = 0.5 * (dFsk[4, ...] - amx * dqsk[4, ...])
    bterm_m = 0.5 * (dFsk[3, ...] - amx * dqsk[3, ...])
    cterm_m = 0.5 * (dFsk[2, ...] - amx * dqsk[2, ...])
    dterm_m = 0.5 * (dFsk[1, ...] - amx * dqsk[1, ...])
    
    IS0_m = 13.0 * (aterm_m - bterm_m)**2 + 3.0 * (aterm_m - 3.0*bterm_m)**2
    IS1_m = 13.0 * (bterm_m - cterm_m)**2 + 3.0 * (bterm_m + cterm_m)**2
    IS2_m = 13.0 * (cterm_m - dterm_m)**2 + 3.0 * (3.0*cterm_m - dterm_m)**2
    
    alpha0_m = 1.0 / (epsilon + IS0_m)**2
    alpha1_m = 6.0 / (epsilon + IS1_m)**2
    alpha2_m = 3.0 / (epsilon + IS2_m)**2
    
    alpha_sum_m = alpha0_m + alpha1_m + alpha2_m
    omega0_m = alpha0_m / alpha_sum_m
    omega2_m = alpha2_m / alpha_sum_m
    
    third = (omega0_m * (aterm_m - 2.0*bterm_m + cterm_m) / 3.0 
             + (omega2_m - 0.5) * (bterm_m - 2.0*cterm_m + dterm_m) / 6.0)
    
    # Combine
    Fs = first - second + third  # Shape: (7, Nx, Ny, Nz)
    
    # Transform back to physical variables
    # R has shape (N_vars, 7, Nx, Ny, Nz)
    dF = jnp.einsum('nmxyz,mxyz->nxyz', R, Fs)  # Shape: (N_vars, Nx, Ny, Nz)
    
    # dF[i] is now the numerical flux at interface i
    # Return this so the integrator can compute: -dtdl*(dF[i] - dF[i-1])
    return dF


@partial(jax.jit, static_argnames=["registered_variables"])
def _weno_flux_y(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    # Transpose to make y the "x" direction
    qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    
    # Swap components
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]
    B_x = qy[registered_variables.magnetic_index.x]
    B_y = qy[registered_variables.magnetic_index.y]
    
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
    qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
    qy = qy.at[registered_variables.magnetic_index.y].set(B_x)
    
    Fy = _weno_flux_x(qy, gamma, registered_variables)
    
    # Transpose back
    Fy = jnp.transpose(Fy, (0, 2, 1, 3))
    
    # Swap components back
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
    # Transpose to make z the "x" direction
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    
    # Swap components
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]
    B_x = qz[registered_variables.magnetic_index.x]
    B_z = qz[registered_variables.magnetic_index.z]
    
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
    qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
    qz = qz.at[registered_variables.magnetic_index.z].set(B_x)
    
    Fz = _weno_flux_x(qz, gamma, registered_variables)
    
    # Transpose back
    Fz = jnp.transpose(Fz, (0, 3, 2, 1))
    
    # Swap components back
    Fmomentum_x = Fz[registered_variables.momentum_index.x]
    Fmomentum_z = Fz[registered_variables.momentum_index.z]
    FB_x = Fz[registered_variables.magnetic_index.x]
    FB_z = Fz[registered_variables.magnetic_index.z]
    
    Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
    Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)
    Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
    Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
    return Fz

# @jax.jit
# def _weno_interpolate(
#     a1, a2, a3, a4
# ):
#     IS0 = 13 * (a1 - a2) ** 2 + 3 * (a1 - 3 * a2) ** 2
#     IS1 = 13 * (a2 - a3) ** 2 + 3 * (a2 + a3) ** 2
#     IS2 = 13 * (a3 - a4) ** 2 + 3 * (3 * a3 - a4) ** 2

#     epsilon = 1e-8

#     C0 = 0.1
#     C1 = 0.6
#     C2 = 0.3

#     alpha0 = C0 / (epsilon + IS0) ** 2
#     alpha1 = C1 / (epsilon + IS1) ** 2
#     alpha2 = C2 / (epsilon + IS2) ** 2

#     alpha_sum = alpha0 + alpha1 + alpha2

#     w0 = alpha0 / alpha_sum
#     w1 = alpha1 / alpha_sum
#     w2 = alpha2 / alpha_sum

#     return 1/3 * w0 * (a1 - 2 * a2 + a3) + 1/6 * (w2 - 1/2) * (a2 - 2 * a3 + a4)


# @partial(jax.jit, static_argnames=["registered_variables"])
# def _weno_flux_x(
#     conserved_state,
#     gamma: Union[float, jnp.ndarray],
#     registered_variables: RegisteredVariables,
# ):
    
#     q = conserved_state

#     lambdas, R, L = _eigen_x(q, gamma, registered_variables)

#     F = _mhd_flux_x(q, gamma, registered_variables)

#     # lambdas is of shape
#     # (7, 32, 32, 32)

#     # L is of shape
#     # (7, 8, 32, 32, 32)

#     # F is of shape
#     # (8, 32, 32, 32)

#     # F_char is of shape
#     # (7, 32, 32, 32)

#     F_char = jnp.einsum('abxyz,bxyz->axyz', L, F)
#     q_char = jnp.einsum('abxyz,bxyz->axyz', L, q)

#     deltaF_char = jnp.roll(F_char, -1, axis=1) - F_char
#     deltaq_char = jnp.roll(q_char, -1, axis=1) - q_char

#     # find the maximum of lambdas
#     # along the x-axis in the stencil
#     # i - 2 <= m <= i + 3 (sliding window)
#     # -> shape (7, 32, 32, 32)
    
#     # Stack shifted versions of lambdas for positions 
#     # i-2, i-1, i, i+1, i+2, i+3
#     # and take maximum along the stacked dimension
#     lambda_stencil = jnp.stack([
#         jnp.roll(lambdas, 2, axis=1),   # i-2
#         jnp.roll(lambdas, 1, axis=1),   # i-1
#         lambdas,                         # i
#         jnp.roll(lambdas, -1, axis=1),  # i+1
#         jnp.roll(lambdas, -2, axis=1),  # i+2
#         jnp.roll(lambdas, -3, axis=1),  # i+3
#     ], axis=0)
    
#     # Take max over the stencil dimension (axis=0)
#     lambda_max = jnp.max(lambda_stencil, axis=0)

#     # Lax-Friedrichs flux splitting
#     F_plus = 0.5 * (deltaF_char + lambda_max * deltaq_char) # F^+_{i+1/2}
#     F_minus = 0.5 * (deltaF_char - lambda_max * deltaq_char) # F^-_{i+1/2}

#     # WENO reconstruction for F_plus
#     phi_plus = _weno_interpolate(
#         jnp.roll(F_plus, 2, axis=1),
#         jnp.roll(F_plus, 1, axis=1),
#         F_plus,
#         jnp.roll(F_plus, -1, axis=1),
#     )

#     # WENO reconstruction for F_minus
#     phi_minus = _weno_interpolate(
#         jnp.roll(F_minus, -2, axis=1),
#         jnp.roll(F_minus, -1, axis=1),
#         F_minus,
#         jnp.roll(F_minus, 1, axis=1),
#     )

#     F_interface_char = phi_minus - phi_plus
#     F_interface = jnp.einsum('abxyz,bxyz->axyz', R, F_interface_char)

#     F_interface = 1/12 * (-jnp.roll(F, 1, axis=1) + 7 * F + 7 * jnp.roll(F, -1, axis=1) - jnp.roll(F, -2, axis=1)) + F_interface

#     return F_interface

# @partial(jax.jit, static_argnames=["registered_variables"])
# def _weno_flux_y(
#     conserved_state,
#     gamma: Union[float, jnp.ndarray],
#     registered_variables: RegisteredVariables,
# ):
#     # conserved_state is of shape (num_variables, nx, ny, nz)
#     # need to permute to bring y-axis to x-axis position
#     # and bring v_y to v_x position and B_y to B_x position
#     qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
#     momentum_x = qy[registered_variables.momentum_index.x]
#     momentum_y = qy[registered_variables.momentum_index.y]
#     B_x = qy[registered_variables.magnetic_index.x]
#     B_y = qy[registered_variables.magnetic_index.y]
#     qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
#     qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
#     qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
#     qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

#     Fy = _weno_flux_x(qy, gamma, registered_variables)

#     # need to permute back
#     Fy = jnp.transpose(Fy, (0, 2, 1, 3))
#     Fmomentum_x = Fy[registered_variables.momentum_index.x]
#     Fmomentum_y = Fy[registered_variables.momentum_index.y]
#     FB_x = Fy[registered_variables.magnetic_index.x]
#     FB_y = Fy[registered_variables.magnetic_index.y]
#     Fy = Fy.at[registered_variables.momentum_index.x].set(Fmomentum_y)
#     Fy = Fy.at[registered_variables.momentum_index.y].set(Fmomentum_x)
#     Fy = Fy.at[registered_variables.magnetic_index.x].set(FB_y)
#     Fy = Fy.at[registered_variables.magnetic_index.y].set(FB_x)
    
#     return Fy

# @partial(jax.jit, static_argnames=["registered_variables"])
# def _weno_flux_z(
#     conserved_state,
#     gamma: Union[float, jnp.ndarray],
#     registered_variables: RegisteredVariables,
# ):
#     # conserved_state is of shape (num_variables, nx, ny, nz)
#     # need to permute to bring z-axis to x-axis position
#     # and bring v_z to v_x position and B_z to B_x position
#     qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
#     momentum_x = qz[registered_variables.momentum_index.x]
#     momentum_z = qz[registered_variables.momentum_index.z]
#     B_x = qz[registered_variables.magnetic_index.x]
#     B_z = qz[registered_variables.magnetic_index.z]
#     qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
#     qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
#     qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
#     qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

#     Fz = _weno_flux_x(qz, gamma, registered_variables)

#     # need to permute back
#     Fz = jnp.transpose(Fz, (0, 3, 2, 1))
#     Fmomentum_x = Fz[registered_variables.momentum_index.x]
#     Fmomentum_z = Fz[registered_variables.momentum_index.z]
#     FB_x = Fz[registered_variables.magnetic_index.x]
#     FB_z = Fz[registered_variables.magnetic_index.z]
#     Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
#     Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)
#     Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
#     Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
#     return Fz