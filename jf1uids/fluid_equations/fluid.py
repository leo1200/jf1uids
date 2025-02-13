from types import NoneType
import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import total_energy_from_primitives_with_crs, total_pressure_from_conserved_with_crs
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import FIELD_TYPE, STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams

# The default state jf1uids operates on 
# are the primitive variables rho, u, p.

# We might want to add more variables to the
# state array, e.g. a tracer for stellar wind
# mass or cosmic ray pressure. 


# ======= Create the primitive state ========

@jaxtyped(typechecker=typechecker)
def construct_primitive_state(

    config: SimulationConfig,
    registered_variables: RegisteredVariables,

    density: FIELD_TYPE,
    velocity_x: Union[FIELD_TYPE, NoneType] = None,
    velocity_y: Union[FIELD_TYPE, NoneType] = None,
    velocity_z: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    gas_pressure: Union[FIELD_TYPE, NoneType] = None,
    cosmic_ray_pressure: Union[FIELD_TYPE, NoneType] = None,

) -> STATE_TYPE:

    """Stack the primitive variables into the state array.

    IN 1D SET ONLY THE XCOMPONENTS, in 2D SET X AND Y COMPONENTS,
    in 3D SET X, Y AND Z COMPONENTS
    
    Args:
        rho: The density of the fluid.
        u: The velocity of the fluid.
        p: The pressure of the fluid.
        registered_variables: The indices of the variables in the state array.
        
    Returns:
        The state array.
    """

    state = jnp.zeros((registered_variables.num_vars, *density.shape))
    state = state.at[registered_variables.density_index].set(density)

    if config.dimensionality == 1:
        state = state.at[registered_variables.velocity_index].set(velocity_x)
    elif config.dimensionality == 2:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
    elif config.dimensionality == 3:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
        state = state.at[registered_variables.velocity_index.z].set(velocity_z)

    if config.mhd:
        if config.dimensionality == 1:
            state = state.at[registered_variables.magnetic_index].set(magnetic_field_x)
        elif config.dimensionality >= 2:
            state = state.at[registered_variables.magnetic_index.x].set(magnetic_field_x)
            state = state.at[registered_variables.magnetic_index.y].set(magnetic_field_y)
            state = state.at[registered_variables.magnetic_index.z].set(magnetic_field_z)

    state = state.at[registered_variables.pressure_index].set(gas_pressure)

    if registered_variables.cosmic_ray_n_active:

        # TODO: get from params
        gamma_cr = 4/3

        state = state.at[registered_variables.pressure_index].set(gas_pressure + cosmic_ray_pressure)
        state = state.at[registered_variables.cosmic_ray_n_index].set(cosmic_ray_pressure ** (1/gamma_cr))

    return state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['registered_variables'])
def construct_primitive_state3D(
    rho: Float[Array, "num_cells num_cells num_cells"],
    u_x: Float[Array, "num_cells num_cells num_cells"],
    u_y: Float[Array, "num_cells num_cells num_cells"],
    u_z: Float[Array, "num_cells num_cells num_cells"],
    p: Float[Array, "num_cells num_cells num_cells"],
    registered_variables: RegisteredVariables
) -> Float[Array, "num_vars num_cells num_cells num_cells"]:
    """Stack the primitive variables into the state array.
    
    Args:
        rho: The density of the fluid.
        u: The velocity of the fluid.
        p: The pressure of the fluid.
        registered_variables: The indices of the variables in the state array.
        
    Returns:
        The state array.
    """

    state = jnp.zeros((registered_variables.num_vars, rho.shape[0], rho.shape[1], rho.shape[2]))
    state = state.at[registered_variables.density_index].set(rho)

    state = state.at[registered_variables.velocity_index.x].set(u_x)
    state = state.at[registered_variables.velocity_index.y].set(u_y)
    state = state.at[registered_variables.velocity_index.z].set(u_z)

    state = state.at[registered_variables.pressure_index].set(p)

    return state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def primitive_state_from_conserved(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> STATE_TYPE:
    """Convert the conserved state to the primitive state.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The primitive state.
    """
    # note the indices of the conserved variables
    # are the same as the indices of the primitive variables
    # so velocity and moentum density have the same index

    rho = conserved_state[registered_variables.density_index]
    E = conserved_state[registered_variables.pressure_index]

    if config.dimensionality == 1:
        u = conserved_state[registered_variables.velocity_index] / rho
    elif config.dimensionality == 2:
        ux = conserved_state[registered_variables.velocity_index.x] / rho
        uy = conserved_state[registered_variables.velocity_index.y] / rho
        u = jnp.sqrt(ux**2 + uy**2)
    elif config.dimensionality == 3:
        ux = conserved_state[registered_variables.velocity_index.x] / rho
        uy = conserved_state[registered_variables.velocity_index.y] / rho
        uz = conserved_state[registered_variables.velocity_index.z] / rho
        u = jnp.sqrt(ux**2 + uy**2 + uz**2)

    p = pressure_from_energy(E, rho, u, gamma)

    if registered_variables.cosmic_ray_n_active:
        p = total_pressure_from_conserved_with_crs(conserved_state, registered_variables)
    else:
        p = pressure_from_energy(E, rho, u, gamma)

    # set the primitive state
    primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

    if config.dimensionality == 1:
        primitive_state = primitive_state.at[registered_variables.velocity_index].set(u)
    elif config.dimensionality == 2:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(ux)
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(uy)
    elif config.dimensionality == 3:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(ux)
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(uy)
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(uz)
    
    # for all other variables assume that primitive and conserved state are the same
    # as for the mass density

    return primitive_state

# ===========================================

# ======= Create the conserved state ========


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def conserved_state_from_primitive(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> STATE_TYPE:
    """Convert the primitive state to the conserved state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index of the fluid.

    Returns:
        The conserved state.
    """
    
    rho = primitive_state[registered_variables.density_index]

    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]

    if registered_variables.cosmic_ray_n_active:
        E = total_energy_from_primitives_with_crs(primitive_state, registered_variables)
    else:
        E = total_energy_from_primitives(rho, u, p, gamma)

    conserved_state = primitive_state.at[registered_variables.pressure_index].set(E)

    if config.dimensionality == 1:
        conserved_state = conserved_state.at[registered_variables.velocity_index].set(rho * primitive_state[registered_variables.velocity_index])
    elif config.dimensionality == 2:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(rho * primitive_state[registered_variables.velocity_index.x])
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(rho * primitive_state[registered_variables.velocity_index.y])
    elif config.dimensionality == 3:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(rho * primitive_state[registered_variables.velocity_index.x])
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(rho * primitive_state[registered_variables.velocity_index.y])
        conserved_state = conserved_state.at[registered_variables.velocity_index.z].set(rho * primitive_state[registered_variables.velocity_index.z])
    else:
        raise ValueError("Invalid dimension.")

    return conserved_state

# ===========================================

# =============== Fluid physics ===============

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def get_absolute_velocity(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables
) -> Union[Float[Array, "num_cells"], Float[Array, "num_cells_x num_cells_y"], Float[Array, "num_cells_x num_cells_y num_cells_z"]]:
    """Get the absolute velocity of the fluid.

    Args:
        primitive_state: The primitive state of the fluid.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The absolute velocity.
    """
    if config.dimensionality == 1:
        return jnp.abs(primitive_state[registered_variables.velocity_index])
    elif config.dimensionality == 2:
        return jnp.sqrt(primitive_state[registered_variables.velocity_index.x]**2 + primitive_state[registered_variables.velocity_index.y]**2)
    elif config.dimensionality == 3:
        return jnp.sqrt(primitive_state[registered_variables.velocity_index.x]**2 + primitive_state[registered_variables.velocity_index.y]**2 + primitive_state[registered_variables.velocity_index.z]**2)
    else:
        raise ValueError("Invalid dimension.")

@jax.jit
def pressure_from_internal_energy(e, rho, gamma):
  """
  Calculate the pressure from the internal energy.
  
  Args:
      e: The internal energy.
      rho: The density.
      gamma: The adiabatic index.
      
  Returns:
      The pressure.
  """
  return (gamma - 1) * rho * e

@jax.jit
def internal_energy_from_energy(E, rho, u):
    """ Calculate the internal energy from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.

    Returns:
        The internal energy.
    """
    return E / rho - 0.5 * u**2

@jax.jit
def pressure_from_energy(E, rho, u, gamma):
    """ Calculate the pressure from the total energy.

    Args:
        E: The total energy.
        rho: The density.
        u: The velocity.
        gamma: The adiabatic index.

    Returns:
        The pressure.
    """

    e = internal_energy_from_energy(E, rho, u)
    return pressure_from_internal_energy(e, rho, gamma)

@jax.jit
def total_energy_from_primitives(rho, u, p, gamma):
    """Calculate the total energy from the primitive variables.

    Args:
        rho: The density.
        u: The velocity.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The total energy.
    """

    return p / (gamma - 1) + 0.5 * rho * u**2

@jax.jit
def speed_of_sound(rho, p, gamma):
    """Calculate the speed of sound.

    Args:
        rho: The density.
        p: The pressure.
        gamma: The adiabatic index.

    Returns:
        The speed of sound.
    """
    return jnp.sqrt(gamma * p / rho)


# ===========================================