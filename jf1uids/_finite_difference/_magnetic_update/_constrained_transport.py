"""
Constrained Transport (CT) implementation for MHD.
Based on HOW-MHD paper (Seo & Ryu 2023).

Algorithm summary
-----------------

We carry interface magnetic fields

b_x at x-interfaces,
b_y at y-interfaces,
b_z at z-interfaces

through the simulation, updating them using the CT
algorithm such that (ignoring floating point errors) the
divergence of B remains zero. This is achieved by updating
the interfaces based on the discrete curl of an electric
field defined at cell edges.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Union
from jaxtyping import Array, Float

from jf1uids.variable_registry.registered_variables import RegisteredVariables

XAXIS = 0
YAXIS = 1
ZAXIS = 2

@partial(jax.jit, static_argnames=["registered_variables", "ct_order"])
def constrained_transport_rhs(
    conserved_state,
    weno_flux_x,
    weno_flux_y,
    weno_flux_z,
    time_step,
    grid_spacing,
    registered_variables: RegisteredVariables,
    ct_order: int = 6,
):
    """
    Here we compute the RHS updates for the interface magnetic fields.
    """

    # Step 0: retrieve variables

    # at index i, the fluxes are at interfaces i+1/2
    By_flux_x_interface = weno_flux_x[registered_variables.magnetic_index.y]
    Bz_flux_x_interface = weno_flux_x[registered_variables.magnetic_index.z]
    Bx_flux_y_interface = weno_flux_y[registered_variables.magnetic_index.x]
    Bz_flux_y_interface = weno_flux_y[registered_variables.magnetic_index.z]
    Bx_flux_z_interface = weno_flux_z[registered_variables.magnetic_index.x]
    By_flux_z_interface = weno_flux_z[registered_variables.magnetic_index.y]

    # cell-centered variables
    rho = conserved_state[registered_variables.density_index]
    vx = conserved_state[registered_variables.momentum_index.x] / rho
    vy = conserved_state[registered_variables.momentum_index.y] / rho
    vz = conserved_state[registered_variables.momentum_index.z] / rho
    Bx = conserved_state[registered_variables.magnetic_index.x]
    By = conserved_state[registered_variables.magnetic_index.y]
    Bz = conserved_state[registered_variables.magnetic_index.z]
    
    # Step 1: Compute modified magnetic field fluxes (Eq. 12 - 17)
    
    # At x-interfaces, after the interpolation at index i, 
    # we have values at i+1/2
    Bx_x_interface = interp4(Bx, XAXIS)
    vy_x_interface = interp4(vy, XAXIS)
    vz_x_interface = interp4(vz, XAXIS)

    By_flux_x_interface_mod = By_flux_x_interface + Bx_x_interface * vy_x_interface
    Bz_flux_x_interface_mod = Bz_flux_x_interface + Bx_x_interface * vz_x_interface

    # At y-interfaces
    By_y_interface = interp4(By, YAXIS)
    vx_y_interface = interp4(vx, YAXIS)
    vz_y_interface = interp4(vz, YAXIS)

    Bx_flux_y_interface_mod = Bx_flux_y_interface + By_y_interface * vx_y_interface
    Bz_flux_y_interface_mod = Bz_flux_y_interface + By_y_interface * vz_y_interface
    
    # At z-interfaces
    Bz_z_interface = interp4(Bz, ZAXIS)
    vx_z_interface = interp4(vx, ZAXIS)
    vy_z_interface = interp4(vy, ZAXIS)

    Bx_flux_z_interface_mod = Bx_flux_z_interface + Bz_z_interface * vx_z_interface
    By_flux_z_interface_mod = By_flux_z_interface + Bz_z_interface * vy_z_interface
    
    # Step 2: Compute electric field components at cell edges (Equations 19-21)
    
    # interpolate from the y interfaces to the (x,y) edges
    g_star_x_edge = interp4(Bx_flux_y_interface_mod, XAXIS)

    # interpolate from the x interfaces to the (x,y) edges
    f_star_y_edge = interp4(By_flux_x_interface_mod, YAXIS)

    # electric field component at (x,y) edges
    Omega_z_edge = g_star_x_edge - f_star_y_edge
    
    # interpolate from the z interfaces to the (y,z) edges
    h_star_y_edge = interp4(By_flux_z_interface_mod, YAXIS)

    # interpolate from the y interfaces to the (y,z) edges
    g_star_z_edge = interp4(Bz_flux_y_interface_mod, ZAXIS)

    # electric field component at (y,z) edges
    Omega_x_edge = h_star_y_edge - g_star_z_edge

    # interpolate from the x interfaces to the (z,x) edges
    f_star_z_edge = interp4(Bz_flux_x_interface_mod, ZAXIS)

    # interpolate from the z interfaces to the (z,x) edges
    h_star_x_edge = interp4(Bx_flux_z_interface_mod, XAXIS)

    # electric field component at (z,x) edges
    Omega_y_edge = f_star_z_edge - h_star_x_edge
    
    # Step 3: Apply smoothing to edge fluxes (Equation 23 in paper)
    Omega_z_bar = smooth_xy_edge(Omega_z_edge)
    Omega_x_bar = smooth_yz_edge(Omega_x_edge)
    Omega_y_bar = smooth_xz_edge(Omega_y_edge)
    
    # Update the interface magnetic fields based on the discrete curl of
    # the electric field at edges (Eq. 24 - 26)

    if ct_order == 6:
        c1, c2, c3 = 75.0/64.0, -25.0/384.0, 3.0/640.0
    else:  # ct_order == 4
        c1, c2, c3 = 9.0/8.0, -1.0/24.0, 0.0

    rhs_bx = - time_step / grid_spacing * finite_difference46(Omega_z_bar, c1, c2, c3, YAXIS) \
             + time_step / grid_spacing * finite_difference46(Omega_y_bar, c1, c2, c3, ZAXIS)

    rhs_by = - time_step / grid_spacing * finite_difference46(Omega_x_bar, c1, c2, c3, ZAXIS) \
             + time_step / grid_spacing * finite_difference46(Omega_z_bar, c1, c2, c3, XAXIS)

    rhs_bz = - time_step / grid_spacing * finite_difference46(Omega_y_bar, c1, c2, c3, XAXIS) \
             + time_step / grid_spacing * finite_difference46(Omega_x_bar, c1, c2, c3, YAXIS)

    return rhs_bx, rhs_by, rhs_bz


# Fourth-order interpolation to interfaces
@partial(jax.jit, static_argnames=["axis"])
def interp4(arr, axis):
    """
    Interpolate to x-interfaces using 4th order
        f_{i+1/2} = (-f_{i-1} + 9f_{i} + 9f_{i+1} - f_{i+2}) / 16
    The i-th array index in the output corresponds to the i+1/2 interface.
    """
    return (-jnp.roll(arr, 1, axis=axis) + 9*arr + 9*jnp.roll(arr, -1, axis=axis) - jnp.roll(arr, -2, axis=axis)) / 16.0


# Smoothing at edges
def smooth_xy_edge(omega):
    """Smooth at (x,y) edges."""
    smooth_x = (jnp.roll(omega, 1, axis=0) - 2*omega + jnp.roll(omega, -1, axis=0)) / 24.0
    smooth_y = (jnp.roll(omega, 1, axis=1) - 2*omega + jnp.roll(omega, -1, axis=1)) / 24.0
    return omega + smooth_x + smooth_y

def smooth_yz_edge(omega):
    """Smooth at (y,z) edges."""
    smooth_y = (jnp.roll(omega, 1, axis=1) - 2*omega + jnp.roll(omega, -1, axis=1)) / 24.0
    smooth_z = (jnp.roll(omega, 1, axis=2) - 2*omega + jnp.roll(omega, -1, axis=2)) / 24.0
    return omega + smooth_y + smooth_z

def smooth_xz_edge(omega):
    """Smooth at (x,z) edges."""
    smooth_x = (jnp.roll(omega, 1, axis=0) - 2*omega + jnp.roll(omega, -1, axis=0)) / 24.0
    smooth_z = (jnp.roll(omega, 1, axis=2) - 2*omega + jnp.roll(omega, -1, axis=2)) / 24.0
    return omega + smooth_x + smooth_z

# Finite difference derivatives
@partial(jax.jit, static_argnames=["axis"])
def finite_difference46(omega, c1, c2, c3, axis):
    """High-order FD derivative in x."""
    return c1 * (omega - jnp.roll(omega, 1, axis=axis)) + \
           c2 * (jnp.roll(omega, -1, axis=axis) - jnp.roll(omega, 2, axis=axis)) + \
           c3 * (jnp.roll(omega, -2, axis=axis) - jnp.roll(omega, 3, axis=axis))

# Interpolation from interfaces back to centers
@partial(jax.jit, static_argnames=["registered_variables"])
def update_cell_center_fields(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    registered_variables: RegisteredVariables,
) -> Float[Array, "8 Nx Ny Nz"]:
    """Update cell-centered B field from interface values using 6th order interpolation."""
    
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z

    b2_old = conserved_state[BX]**2 + conserved_state[BY]**2 + conserved_state[BZ]**2
    
    # Sixth-order interpolation (Equation 30)
    Bx_center = (3*jnp.roll(bx_interface, 3, axis=0) - 25*jnp.roll(bx_interface, 2, axis=0) + 
                 150*jnp.roll(bx_interface, 1, axis=0) + 150*bx_interface - 
                 25*jnp.roll(bx_interface, -1, axis=0) + 3*jnp.roll(bx_interface, -2, axis=0)) / 256.0
    
    By_center = (3*jnp.roll(by_interface, 3, axis=1) - 25*jnp.roll(by_interface, 2, axis=1) + 
                 150*jnp.roll(by_interface, 1, axis=1) + 150*by_interface - 
                 25*jnp.roll(by_interface, -1, axis=1) + 3*jnp.roll(by_interface, -2, axis=1)) / 256.0
    
    Bz_center = (3*jnp.roll(bz_interface, 3, axis=2) - 25*jnp.roll(bz_interface, 2, axis=2) + 
                 150*jnp.roll(bz_interface, 1, axis=2) + 150*bz_interface - 
                 25*jnp.roll(bz_interface, -1, axis=2) + 3*jnp.roll(bz_interface, -2, axis=2)) / 256.0
    
    conserved_new = conserved_state.at[BX].set(Bx_center)
    conserved_new = conserved_new.at[BY].set(By_center)
    conserved_new = conserved_new.at[BZ].set(Bz_center)

    b2_new = conserved_new[BX]**2 + conserved_new[BY]**2 + conserved_new[BZ]**2

    # update total energy: E_new = E_old + 0.5 * (b2_new - b2_old)
    conserved_new = conserved_new.at[registered_variables.pressure_index].add(0.5 * (b2_new - b2_old))
    
    return conserved_new


@partial(jax.jit, static_argnames=["registered_variables"])
def initialize_interface_fields(
    conserved_state,
    registered_variables: RegisteredVariables,
):
    """Initialize magnetic field at interfaces from cell centers."""
    
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    
    # Use fourth-order interpolation
    bx_interface = interp4(Bx, XAXIS)
    by_interface = interp4(By, YAXIS)
    bz_interface = interp4(Bz, ZAXIS)

    return bx_interface, by_interface, bz_interface