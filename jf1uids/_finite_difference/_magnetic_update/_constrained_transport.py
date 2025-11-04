"""
Corrected Constrained Transport (CT) implementation for MHD.
Based on HOW-MHD paper (Seo & Ryu 2023).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Union
from jaxtyping import Array, Float

from jf1uids.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["registered_variables", "ct_order"])
def constrained_transport_rhs(
    conserved_state: Float[Array, "8 Nx Ny Nz"],
    bx_interface: Float[Array, "Nx+1 Ny Nz"],  
    by_interface: Float[Array, "Nx Ny+1 Nz"],  
    bz_interface: Float[Array, "Nx Ny Nz+1"],  
    flux_x: Float[Array, "8 Nx+1 Ny Nz"],  # Full flux vector in x-direction
    flux_y: Float[Array, "8 Nx Ny+1 Nz"],  # Full flux vector in y-direction
    flux_z: Float[Array, "8 Nx Ny Nz+1"],  # Full flux vector in z-direction
    dtdx: float,
    dtdy: float,
    dtdz: float,
    registered_variables: RegisteredVariables,
    ct_order: int = 6,
) -> Tuple[Float[Array, "Nx+1 Ny Nz"], Float[Array, "Nx Ny+1 Nz"], Float[Array, "Nx Ny Nz+1"]]:
    """
    Update magnetic field at interfaces using constrained transport.
    
    Returns:
        Updated interface magnetic fields (bx_new, by_new, bz_new)
    """
    
    # Get indices
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    
    # Extract magnetic fluxes from the full flux vectors
    # These are already at the correct interfaces
    f_y = flux_x[BY]  # By flux at x-interfaces
    f_z = flux_x[BZ]  # Bz flux at x-interfaces
    g_x = flux_y[BX]  # Bx flux at y-interfaces  
    g_z = flux_y[BZ]  # Bz flux at y-interfaces
    h_x = flux_z[BX]  # Bx flux at z-interfaces
    h_y = flux_z[BY]  # By flux at z-interfaces
    
    # Get velocity and B field for modified fluxes
    rho = conserved_state[DI]
    vx = conserved_state[MX] / rho
    vy = conserved_state[MY] / rho
    vz = conserved_state[MZ] / rho
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    
    # Step 1: Compute modified magnetic field fluxes (Equations 12-17)
    # The WENO fluxes already include the advection terms, but we need to add
    # the additional terms for the CT algorithm
    
    # At x-interfaces
    Bx_xi = interp4_to_xface(Bx)
    vy_xi = interp4_to_xface(vy)
    vz_xi = interp4_to_xface(vz)
    
    f_star_y = f_y + Bx_xi * vy_xi  # Modified By flux
    f_star_z = f_z + Bx_xi * vz_xi  # Modified Bz flux
    
    # At y-interfaces  
    By_yi = interp4_to_yface(By)
    vx_yi = interp4_to_yface(vx)
    vz_yi = interp4_to_yface(vz)
    
    g_star_x = g_x + By_yi * vx_yi  # Modified Bx flux
    g_star_z = g_z + By_yi * vz_yi  # Modified Bz flux
    
    # At z-interfaces
    Bz_zi = interp4_to_zface(Bz)
    vx_zi = interp4_to_zface(vx)
    vy_zi = interp4_to_zface(vy)
    
    h_star_x = h_x + Bz_zi * vx_zi  # Modified Bx flux
    h_star_y = h_y + Bz_zi * vy_zi  # Modified By flux
    
    # Step 2: Compute electric field components at cell edges (Equations 19-21)
    # Omega_z at (x,y) edges: needs interpolation from interfaces to edges
    g_star_x_edge = interp4_to_xy_edge_from_y(g_star_x)  
    f_star_y_edge = interp4_to_xy_edge_from_x(f_star_y)  
    Omega_z_edge = g_star_x_edge - f_star_y_edge
    
    # Omega_x at (y,z) edges
    h_star_y_edge = interp4_to_yz_edge_from_z(h_star_y)
    g_star_z_edge = interp4_to_yz_edge_from_y(g_star_z)
    Omega_x_edge = h_star_y_edge - g_star_z_edge
    
    # Omega_y at (x,z) edges  
    f_star_z_edge = interp4_to_xz_edge_from_x(f_star_z)
    h_star_x_edge = interp4_to_xz_edge_from_z(h_star_x)
    Omega_y_edge = f_star_z_edge - h_star_x_edge
    
    # Step 3: Apply smoothing to edge fluxes (Equation 23 in paper)
    Omega_z_bar = smooth_xy_edge(Omega_z_edge)
    Omega_x_bar = smooth_yz_edge(Omega_x_edge)
    Omega_y_bar = smooth_xz_edge(Omega_y_edge)
    
    # Step 4: Update interface magnetic fields using high-order FD
    if ct_order == 6:
        c1, c2, c3 = 75.0/64.0, -25.0/384.0, 3.0/640.0
    else:  # ct_order == 4
        c1, c2, c3 = 9.0/8.0, -1.0/24.0, 0.0
    
    # Update bx at x-interfaces (Equation 24)
    # dbx/dt = -dOz/dy + dOy/dz
    rhs_bx = - dtdy * fd_deriv_y(Omega_z_bar, c1, c2, c3) \
                          + dtdz * fd_deriv_z(Omega_y_bar, c1, c2, c3)
    
    # Update by at y-interfaces (Equation 25)
    # dby/dt = -dOx/dz + dOz/dx
    rhs_by = - dtdz * fd_deriv_z(Omega_x_bar, c1, c2, c3) \
                          + dtdx * fd_deriv_x(Omega_z_bar, c1, c2, c3)
    
    # Update bz at z-interfaces (Equation 26)
    # dbz/dt = -dOy/dx + dOx/dy
    rhs_bz = - dtdx * fd_deriv_x(Omega_y_bar, c1, c2, c3) \
                          + dtdy * fd_deriv_y(Omega_x_bar, c1, c2, c3)

    return rhs_bx, rhs_by, rhs_bz


# Fourth-order interpolation to interfaces
def interp4_to_xface(arr):
    """Interpolate to x-interfaces using 4th order."""
    # From cell centers to x-faces (between i and i+1)
    return (-jnp.roll(arr, 1, axis=0) + 9*arr + 9*jnp.roll(arr, -1, axis=0) - jnp.roll(arr, -2, axis=0)) / 16.0

def interp4_to_yface(arr):
    """Interpolate to y-interfaces using 4th order."""
    return (-jnp.roll(arr, 1, axis=1) + 9*arr + 9*jnp.roll(arr, -1, axis=1) - jnp.roll(arr, -2, axis=1)) / 16.0

def interp4_to_zface(arr):
    """Interpolate to z-interfaces using 4th order."""
    return (-jnp.roll(arr, 1, axis=2) + 9*arr + 9*jnp.roll(arr, -1, axis=2) - jnp.roll(arr, -2, axis=2)) / 16.0

# Interpolation to edges
def interp4_to_xy_edge_from_x(arr):
    """From x-interfaces to (x,y) edges."""
    return interp4_to_yface(arr)

def interp4_to_xy_edge_from_y(arr):
    """From y-interfaces to (x,y) edges."""
    return interp4_to_xface(arr)

def interp4_to_yz_edge_from_y(arr):
    """From y-interfaces to (y,z) edges."""
    return interp4_to_zface(arr)

def interp4_to_yz_edge_from_z(arr):
    """From z-interfaces to (y,z) edges."""
    return interp4_to_yface(arr)

def interp4_to_xz_edge_from_x(arr):
    """From x-interfaces to (x,z) edges."""
    return interp4_to_zface(arr)

def interp4_to_xz_edge_from_z(arr):
    """From z-interfaces to (x,z) edges."""
    return interp4_to_xface(arr)

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
def fd_deriv_x(omega, c1, c2, c3):
    """High-order FD derivative in x."""
    return c1 * (omega - jnp.roll(omega, 1, axis=0)) + \
           c2 * (jnp.roll(omega, -1, axis=0) - jnp.roll(omega, 2, axis=0)) + \
           c3 * (jnp.roll(omega, -2, axis=0) - jnp.roll(omega, 3, axis=0))

def fd_deriv_y(omega, c1, c2, c3):
    """High-order FD derivative in y."""
    return c1 * (omega - jnp.roll(omega, 1, axis=1)) + \
           c2 * (jnp.roll(omega, -1, axis=1) - jnp.roll(omega, 2, axis=1)) + \
           c3 * (jnp.roll(omega, -2, axis=1) - jnp.roll(omega, 3, axis=1))

def fd_deriv_z(omega, c1, c2, c3):
    """High-order FD derivative in z."""
    return c1 * (omega - jnp.roll(omega, 1, axis=2)) + \
           c2 * (jnp.roll(omega, -1, axis=2) - jnp.roll(omega, 2, axis=2)) + \
           c3 * (jnp.roll(omega, -2, axis=2) - jnp.roll(omega, 3, axis=2))

# Interpolation from interfaces back to centers
@partial(jax.jit, static_argnames=["registered_variables"])
def update_cell_center_fields(
    conserved_state: Float[Array, "8 Nx Ny Nz"],
    bx_interface: Float[Array, "Nx+1 Ny Nz"],
    by_interface: Float[Array, "Nx Ny+1 Nz"],
    bz_interface: Float[Array, "Nx Ny Nz+1"],
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
    conserved_state: Float[Array, "8 Nx Ny Nz"],
    registered_variables: RegisteredVariables,
) -> Tuple[Float[Array, "Nx+1 Ny Nz"], Float[Array, "Nx Ny+1 Nz"], Float[Array, "Nx Ny Nz+1"]]:
    """Initialize magnetic field at interfaces from cell centers."""
    
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    
    # Use fourth-order interpolation
    bx_interface = interp4_to_xface(Bx)
    by_interface = interp4_to_yface(By)
    bz_interface = interp4_to_zface(Bz)
    
    return bx_interface, by_interface, bz_interface