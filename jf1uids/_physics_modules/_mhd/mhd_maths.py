from functools import partial
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
import jax

from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import pressure_from_energy, total_energy_from_primitives
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# runtime debugging
from jax.experimental import checkify

@jax.jit
def curl2D(field, dx: float):
    """Calculate the curl of a 3d field on a 2d grid.
    assumes dx = dy
    
    Args:
        field: The field to calculate the curl of.
        dx: The width of the cells.
    
    Returns:
        The curl of the field.
    """
    curl = jnp.zeros_like(field)

    curl = curl.at[0, 1:-1, 1:-1].add(0.5 * (field[2, 1:-1, 2:] - field[2, 1:-1, :-2]) / dx)
    curl = curl.at[1, 1:-1, 1:-1].add(-0.5 * (field[2, 2:, 1:-1] - field[2, :-2, 1:-1]) / dx)

    curl = curl.at[2, 1:-1, 1:-1].add(0.5 * (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / dx)
    curl = curl.at[2, 1:-1, 1:-1].add(-0.5 * (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / dx)

    return curl

@jax.jit
def curl3D(field, dx: float):
    """Calculate the curl of a 3d field on a 3d grid.
    
    Args:
        field: The field to calculate the curl of.
        dx: The width of the cells.
    
    Returns:
        The curl of the field.
    """
    curl = jnp.zeros_like(field)

    curl = curl.at[0, :, 1:-1, :].add(0.5 * (field[2, :, 2:, :] - field[2, :, :-2, :]) / dx)
    curl = curl.at[0, :, :, 1:-1].add(-0.5 * (field[1, :, :, 2:] - field[1, :, :, :-2]) / dx)

    curl = curl.at[1, :, :, 1:-1].add(0.5 * (field[0, :, :, 2:] - field[0, :, :, :-2]) / dx)
    curl = curl.at[1, 1:-1, :, :].add(-0.5 * (field[2, 2:, :, :] - field[2, :-2, :, :]) / dx)

    curl = curl.at[2, 1:-1, :, :].add(0.5 * (field[1, 2:, :, :] - field[1, :-2, :, :]) / dx)
    curl = curl.at[2, :, 1:-1, :].add(-0.5 * (field[0, :, 2:, :] - field[0, :, :-2, :]) / dx)

    return curl

@jax.jit
def divergence2D(field, dx: float):
    """Calculate the divergence of a 3d field on a 2d grid.
    
    Args:
        field: The field to calculate the divergence of.
        dx: The width of the cells.
    
    Returns:
        The divergence of the field.
    """
    divergence = jnp.zeros_like(field)

    divergence = divergence.at[0, 1:-1, 1:-1].add((field[0, 2:, 1:-1] - field[0, :-2, 1:-1]) / (2 * dx))
    divergence = divergence.at[1, 1:-1, 1:-1].add((field[1, 1:-1, 2:] - field[1, 1:-1, :-2]) / (2 * dx))

    divergence = jnp.sum(divergence, axis = 0)

    return divergence

@jax.jit
def cross(a, b):
    result = jnp.zeros_like(a)
    result = result.at[0, ...].set(a[1, ...] * b[2, ...] - a[2, ...] * b[1, ...])
    result = result.at[1, ...].set(a[2, ...] * b[0, ...] - a[0, ...] * b[2, ...])
    result = result.at[2, ...].set(a[0, ...] * b[1, ...] - a[1, ...] * b[0, ...])
    return result

@partial(jax.jit, static_argnames=['registered_variables', 'config'])
def magnetic_update(magnetic_field, gas_state, dx, dt, registered_variables, config):
    """Update the magnetic field and gas state.
    
    Args:
        magnetic_field: The magnetic field.
        gas_state: The gas state.
        dx: The width of the cells.
        dt: The time step.
    
    Returns:
        The updated magnetic field.
    """

    # retrieve the velocity
    velocity = jnp.zeros((3, *gas_state.shape[1:]), dtype = jnp.float64)
    velocity = velocity.at[0:2, :, :].set(gas_state[registered_variables.velocity_index.x:registered_variables.velocity_index.x + 2, ...])

    density = gas_state[registered_variables.density_index, ...]

    def phi(velocity, magnetic_field):
        # calculate the electric field, electric field = -v x B
        electric_field = cross(magnetic_field, velocity)

        # calculate the curl of the electric field
        if config.dimensionality == 2:
            curl_electric_field = curl2D(electric_field, dx)
        elif config.dimensionality == 3:
            curl_electric_field = curl3D(electric_field, dx)
        else:
            raise ValueError("No 1D curl.")

        # calculate the curl of the magnetic field
        if config.dimensionality == 2:
            curl_magnetic_field = curl2D(magnetic_field, dx)
        elif config.dimensionality == 3:
            curl_magnetic_field = curl3D(magnetic_field, dx)
        else:
            raise ValueError("No 1D curl.")

        phi1 = curl_electric_field
        phi2 = cross(magnetic_field, curl_magnetic_field) / density

        return phi1, phi2
    
    magnetic_field = _boundary_handler(magnetic_field, config)
    velocity = _boundary_handler(velocity, config)
    
    B_0 = magnetic_field
    v_0 = velocity

    avg_B = (B_0 + magnetic_field) / 2
    avg_v = (v_0 + velocity) / 2

    phiA, phiB = phi(avg_v, avg_B)

    B_1 = magnetic_field - dt * phiA
    v_1 = velocity - dt * phiB

    B_1 = _boundary_handler(B_1, config)
    v_1 = _boundary_handler(v_1, config)

    def while_condition(state):
        B_k, v_k, B_kp1, v_kp1, current_iter = state
        max_change = jnp.maximum(
            jnp.max(jnp.linalg.norm(B_k - B_kp1, axis=0)),
            jnp.max(jnp.linalg.norm(v_k - v_kp1, axis=0))
        )
        return (max_change > 1e-10) & (current_iter < 1000)

    def while_body(state):
        B_k, v_k, B_kp1, v_kp1, current_iter = state

        B_k = B_kp1
        v_k = v_kp1

        avg_B = (B_k + magnetic_field) / 2
        avg_v = (v_k + velocity) / 2

        phiA, phiB = phi(avg_v, avg_B)

        B_kp1 = magnetic_field - dt * phiA
        v_kp1 = velocity - dt * phiB
        
        B_kp1 = _boundary_handler(B_kp1, config)
        v_kp1 = _boundary_handler(v_kp1, config)

        return B_k, v_k, B_kp1, v_kp1, current_iter + 1
    
    _, _, B_n, v_n, _ = jax.lax.while_loop(while_condition, while_body, (B_0, v_0, B_1, v_1, 0))
    
    if config.runtime_debugging:

        avg_B = (B_n + magnetic_field) / 2
        avg_v = (v_n + velocity) / 2
        
        phiA, phiB = phi(avg_v, avg_B)

        B_n_mod = magnetic_field - dt * phiA
        v_n_mod = velocity - dt * phiB

        checkify.check(jnp.sum(jnp.linalg.norm(B_n - B_n_mod, axis = 0)) < 1e-8, "Eigeniteration for B not converged!")
        checkify.check(jnp.sum(jnp.linalg.norm(v_n - v_n_mod, axis = 0)) < 1e-8, "Eigeniteration for v not converged!")

        checkify.check(jnp.all(jnp.abs(divergence2D(B_n, dx) - divergence2D(B_0, dx)) < 1e-8), "Divergence of magnetic field not conserved, by {divergence}", divergence=jnp.max(jnp.abs(divergence2D(B_n, dx) - divergence2D(B_0, dx))))

    # update the gas energy
    gas_energy = total_energy_from_primitives(density, jnp.linalg.norm(velocity, axis = 0), gas_state[registered_variables.pressure_index, ...], 5/3)
    gas_energy_updated = gas_energy - 0.5 * density * jnp.linalg.norm(velocity, axis = 0)**2 + 0.5 * density * jnp.linalg.norm(v_n, axis = 0)**2
    pressure_updated = pressure_from_energy(gas_energy_updated, density, jnp.linalg.norm(v_n, axis = 0), 5/3)
    
    # update the gas state
    gas_state = gas_state.at[registered_variables.velocity_index.x:registered_variables.velocity_index.x + 2, ...].set(v_n[:2, ...])
    gas_state = gas_state.at[registered_variables.pressure_index, ...].set(pressure_updated)

    return B_n, gas_state