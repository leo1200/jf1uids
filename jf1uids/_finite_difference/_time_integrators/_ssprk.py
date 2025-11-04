from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Tuple

from jf1uids._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from jf1uids._finite_difference._magnetic_update._constrained_transport import constrained_transport_rhs, update_cell_center_fields
from jf1uids.variable_registry.registered_variables import RegisteredVariables


@partial(jax.jit, static_argnames=["registered_variables"])
def _ssprk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method with Constrained Transport (CT).
    
    This implementation follows the logic of the provided Fortran 'ssprk' subroutine.
    """

    # jax.debug.print("Starting SSPRK4 with CT step")

    def compute_rhs(current_q, current_bx, current_by, current_bz, k2_coeff):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """
        dtdx = k2_coeff * dt / grid_spacing
        dtdy = k2_coeff * dt / grid_spacing
        dtdz = k2_coeff * dt / grid_spacing
        
        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(current_q, gamma, registered_variables)
        dF_y = _weno_flux_y(current_q, gamma, registered_variables)
        dF_z = _weno_flux_z(current_q, gamma, registered_variables)

        # Calculate RHS for interface magnetic fields using Constrained Transport
        rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
            current_q,
            current_bx, current_by, current_bz,
            dF_x, dF_y, dF_z,
            dtdx, dtdy, dtdz,
            registered_variables,
            ct_order=6
        )
        
        # Calculate RHS for conserved fluid variables
        rhs_q = -dtdx * (
            (dF_x - jnp.roll(dF_x, 1, axis=1))
            + (dF_y - jnp.roll(dF_y, 1, axis=2))
            + (dF_z - jnp.roll(dF_z, 1, axis=3))
        )
        
        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # Store the initial state (t = n)
    q0 = conserved_state
    bx0, by0, bz0 = bx_interface, by_interface, bz_interface

    # --- Stage 1 ---
    k1_1 = 1.0
    k2_1 = 0.39175222700392
    # k3_1 = 0.0, so it's omitted

    rhs_q0, rhs_bx0, rhs_by0, rhs_bz0 = compute_rhs(q0, bx0, by0, bz0, k2_1)
    
    q1 = q0 + rhs_q0
    bx1, by1, bz1 = bx0 + rhs_bx0, by0 + rhs_by0, bz0 + rhs_bz0

    # --- Stage 2 ---
    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266
    
    rhs_q1, rhs_bx1, rhs_by1, rhs_bz1 = compute_rhs(q1, bx1, by1, bz1, k2_2)

    q2 = k1_2 * q0 + k3_2 * q1 + rhs_q1
    bx2 = k1_2 * bx0 + k3_2 * bx1 + rhs_bx1
    by2 = k1_2 * by0 + k3_2 * by1 + rhs_by1
    bz2 = k1_2 * bz0 + k3_2 * bz1 + rhs_bz1

    # --- Stage 3 ---
    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    rhs_q2, rhs_bx2, rhs_by2, rhs_bz2 = compute_rhs(q2, bx2, by2, bz2, k2_3)
    
    q3 = k1_3 * q0 + k3_3 * q2 + rhs_q2
    bx3 = k1_3 * bx0 + k3_3 * bx2 + rhs_bx2
    by3 = k1_3 * by0 + k3_3 * by2 + rhs_by2
    bz3 = k1_3 * bz0 + k3_3 * bz2 + rhs_bz2

    # --- Stage 4 ---
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227
    
    rhs_q3, rhs_bx3, rhs_by3, rhs_bz3 = compute_rhs(q3, bx3, by3, bz3, k2_4)
    
    q4 = k1_4 * q0 + k3_4 * q3 + rhs_q3
    bx4 = k1_4 * bx0 + k3_4 * bx3 + rhs_bx3
    by4 = k1_4 * by0 + k3_4 * by3 + rhs_by3
    bz4 = k1_4 * bz0 + k3_4 * bz3 + rhs_bz3

    # --- Stage 5 (Final Stage) ---
    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01  # This corresponds to k3 in the Fortran code
    k4_5 = 0.51723167208978     # This corresponds to k4
    k5_5 = -6.518979800418380e-12 # This corresponds to k5

    rhs_q4, rhs_bx4, rhs_by4, rhs_bz4 = compute_rhs(q4, bx4, by4, bz4, k2_5)

    # The Fortran update logic for the 5th stage is:
    # q5_init = q0 + (k4/k1)*q2 + (k5/k1)*q3
    # q5_final = k1*q5_init + k3*q4 + RHS(q4)
    # This simplifies to: q5_final = k1*q0 + k4*q2 + k5*q3 + k3*q4 + RHS(q4)
    q5 = (k1_5 * q0) + (k4_5 * q2) + (k5_5 * q3) + (k3_5 * q4) + rhs_q4
    bx_final = (k1_5 * bx0) + (k4_5 * bx2) + (k5_5 * bx3) + (k3_5 * bx4) + rhs_bx4
    by_final = (k1_5 * by0) + (k4_5 * by2) + (k5_5 * by3) + (k3_5 * by4) + rhs_by4
    bz_final = (k1_5 * bz0) + (k4_5 * bz2) + (k5_5 * bz3) + (k3_5 * bz4) + rhs_bz4

    # Update the cell-centered magnetic fields in the conserved state array
    # from the final interface magnetic fields.
    q_final = update_cell_center_fields(
        q5, bx_final, by_final, bz_final, registered_variables
    )
    
    return q_final, bx_final, by_final, bz_final

# @partial(jax.jit, static_argnames=["registered_variables", "use_ct", "ct_order"])
# def _ssprk4_with_ct(
#     conserved_state,
#     bx_interface,
#     by_interface, 
#     bz_interface,
#     gamma: Union[float, jnp.ndarray],
#     grid_spacing: Union[float, jnp.ndarray],
#     dt: Union[float, jnp.ndarray],
#     registered_variables: RegisteredVariables,
#     use_ct: bool = True,
#     ct_order: int = 6,
# ):
#     """
#     SSPRK4 with Constrained Transport for MHD.
    
#     Key insight: The interface magnetic fields evolve separately from the 
#     cell-centered conserved variables through the CT algorithm.
#     """
    
#     def compute_stage(q_curr, bx_curr, by_curr, bz_curr, k2_coeff):
#         """Compute one RK stage with CT."""
#         dtdl = k2_coeff * dt / grid_spacing
        
#         # Get WENO fluxes
#         flux_x = _weno_flux_x(q_curr, gamma, registered_variables)
#         flux_y = _weno_flux_y(q_curr, gamma, registered_variables)
#         flux_z = _weno_flux_z(q_curr, gamma, registered_variables)
        
#         # Update non-magnetic variables using flux divergence
#         rhs = -dtdl * (
#             (flux_x - jnp.roll(flux_x, 1, axis=1))
#             + (flux_y - jnp.roll(flux_y, 1, axis=2))
#             + (flux_z - jnp.roll(flux_z, 1, axis=3))
#         )
        
#         if use_ct:
#             # Update interface magnetic fields using CT
#             bx_new, by_new, bz_new = constrained_transport_update(
#                 q_curr,
#                 bx_curr, by_curr, bz_curr,
#                 flux_x, flux_y, flux_z,
#                 dtdl, dtdl, dtdl,
#                 registered_variables,
#                 ct_order
#             )
            
#             # Don't update magnetic field through flux divergence - it's handled by CT
#             BX = registered_variables.magnetic_index.x
#             BY = registered_variables.magnetic_index.y
#             BZ = registered_variables.magnetic_index.z
            
#             rhs = rhs.at[BX].set(0.0)
#             rhs = rhs.at[BY].set(0.0)
#             rhs = rhs.at[BZ].set(0.0)
            
#             return rhs, bx_new, by_new, bz_new
#         else:
#             return rhs, bx_curr, by_curr, bz_curr
    
#     # Initial state
#     q0 = conserved_state
#     bx0, by0, bz0 = bx_interface, by_interface, bz_interface
    
#     # Keep copies for Stage 5 special initialization
#     q_init = q0
#     bx_init, by_init, bz_init = bx0, by0, bz0
    
#     # Stage 1
#     k1_1 = 1.0
#     k2_1 = 0.39175222700392
#     k3_1 = 0.0
    
#     rhs1, bx1, by1, bz1 = compute_stage(q0, bx0, by0, bz0, k2_1)
#     q1 = k1_1 * q0 + rhs1
#     if use_ct:
#         q1 = update_cell_center_fields(q1, bx1, by1, bz1, registered_variables)
    
#     # Keep for Stage 5
#     q1_saved = q1
#     bx1_saved, by1_saved, bz1_saved = bx1, by1, bz1
    
#     # Stage 2
#     k1_2 = 0.44437049406734
#     k2_2 = 0.36841059262959
#     k3_2 = 0.55562950593266
    
#     rhs2, bx2, by2, bz2 = compute_stage(q1, bx1, by1, bz1, k2_2)
#     q2 = k1_2 * q_init + k3_2 * q1 + rhs2
#     # For interface fields, apply RK coefficients
#     bx2 = k1_2 * bx_init + k3_2 * bx1_saved + (bx2 - bx1)
#     by2 = k1_2 * by_init + k3_2 * by1_saved + (by2 - by1)
#     bz2 = k1_2 * bz_init + k3_2 * bz1_saved + (bz2 - bz1)
#     if use_ct:
#         q2 = update_cell_center_fields(q2, bx2, by2, bz2, registered_variables)
    
#     # Keep for Stage 5
#     q2_saved = q2
#     bx2_saved, by2_saved, bz2_saved = bx2, by2, bz2
    
#     # Stage 3
#     k1_3 = 0.62010185138540
#     k2_3 = 0.25189177424738
#     k3_3 = 0.37989814861460
    
#     rhs3, bx3, by3, bz3 = compute_stage(q2, bx2, by2, bz2, k2_3)
#     q3 = k1_3 * q_init + k3_3 * q2 + rhs3
#     bx3 = k1_3 * bx_init + k3_3 * bx2_saved + (bx3 - bx2)
#     by3 = k1_3 * by_init + k3_3 * by2_saved + (by3 - by2)
#     bz3 = k1_3 * bz_init + k3_3 * bz2_saved + (bz3 - bz2)
#     if use_ct:
#         q3 = update_cell_center_fields(q3, bx3, by3, bz3, registered_variables)
    
#     # Keep for Stage 5
#     q3_saved = q3
#     bx3_saved, by3_saved, bz3_saved = bx3, by3, bz3
    
#     # Stage 4
#     k1_4 = 0.17807995410773
#     k2_4 = 0.54497475021237
#     k3_4 = 0.82192004589227
    
#     rhs4, bx4, by4, bz4 = compute_stage(q3, bx3, by3, bz3, k2_4)
#     q4 = k1_4 * q_init + k3_4 * q3 + rhs4
#     bx4 = k1_4 * bx_init + k3_4 * bx3_saved + (bx4 - bx3)
#     by4 = k1_4 * by_init + k3_4 * by3_saved + (by4 - by3)
#     bz4 = k1_4 * bz_init + k3_4 * bz3_saved + (bz4 - bz3)
#     if use_ct:
#         q4 = update_cell_center_fields(q4, bx4, by4, bz4, registered_variables)
    
#     # Stage 5 - special initialization
#     k1_5 = -2.081261929715610e-02
#     k2_5 = 0.22600748319395
#     k3_5 = 5.03580947213895e-01
#     k4_5 = 0.51723167208978
#     k5_5 = -6.518979800418380e-12
    
#     # Initialize stage 5
#     q5_init = q_init + (k4_5 / k1_5) * q2_saved + (k5_5 / k1_5) * q3_saved
#     # --- START OF CORRECTION ---
#     # The initialization for b-fields must match the q-field and Fortran logic.
#     # We use the full states (e.g., bx2_saved), not the increments.
#     bx5_init = bx_init + (k4_5 / k1_5) * bx2_saved + (k5_5 / k1_5) * bx3_saved
#     by5_init = by_init + (k4_5 / k1_5) * by2_saved + (k5_5 / k1_5) * by3_saved
#     bz5_init = bz_init + (k4_5 / k1_5) * bz2_saved + (k5_5 / k1_5) * bz3_saved
#     # --- END OF CORRECTION ---
    
#     rhs5, bx5, by5, bz5 = compute_stage(q4, bx4, by4, bz4, k2_5)
    
#     q5 = k1_5 * q5_init + k3_5 * q4 + rhs5
#     bx_final = k1_5 * bx5_init + k3_5 * bx4 + (bx5 - bx4)
#     by_final = k1_5 * by5_init + k3_5 * by4 + (by5 - by4)
#     bz_final = k1_5 * bz5_init + k3_5 * bz4 + (bz5 - bz4)
    
#     if use_ct:
#         q5 = update_cell_center_fields(q5, bx_final, by_final, bz_final, registered_variables)
    
#     return q5, bx_final, by_final, bz_final