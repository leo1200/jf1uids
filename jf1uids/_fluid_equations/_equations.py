import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import (
    total_energy_from_primitives_with_crs,
    total_pressure_from_conserved_with_crs,
)
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)
from jf1uids.option_classes.simulation_params import SimulationParams

# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def primitive_state_from_conserved(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
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
        u = jnp.sqrt(ux**2 + uy**2 + uz**2 + 1e-20)

    p = pressure_from_energy(E, rho, u, gamma)

    if registered_variables.cosmic_ray_n_active:
        p = total_pressure_from_conserved_with_crs(
            conserved_state, registered_variables
        )
    else:
        p = pressure_from_energy(E, rho, u, gamma)

    # set the primitive state
    primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

    if config.dimensionality == 1:
        primitive_state = primitive_state.at[registered_variables.velocity_index].set(u)
    elif config.dimensionality == 2:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            ux
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            uy
        )
    elif config.dimensionality == 3:
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(
            ux
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(
            uy
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(
            uz
        )

    # for all other variables assume that primitive and conserved state are the same
    # as for the mass density

    return primitive_state


# ===========================================

# ======= Create the conserved state ========


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def conserved_state_from_primitive(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
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
        conserved_state = conserved_state.at[registered_variables.velocity_index].set(
            rho * primitive_state[registered_variables.velocity_index]
        )
    elif config.dimensionality == 2:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(
            rho * primitive_state[registered_variables.velocity_index.x]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(
            rho * primitive_state[registered_variables.velocity_index.y]
        )
    elif config.dimensionality == 3:
        conserved_state = conserved_state.at[registered_variables.velocity_index.x].set(
            rho * primitive_state[registered_variables.velocity_index.x]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.y].set(
            rho * primitive_state[registered_variables.velocity_index.y]
        )
        conserved_state = conserved_state.at[registered_variables.velocity_index.z].set(
            rho * primitive_state[registered_variables.velocity_index.z]
        )
    else:
        raise ValueError("Invalid dimension.")

    return conserved_state

@partial(jax.jit, static_argnames=["registered_variables"])
def _u_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    return primitive_state[registered_variables.velocity_index.x]**2 + \
           primitive_state[registered_variables.velocity_index.y]**2 + \
           primitive_state[registered_variables.velocity_index.z]**2

def _b_squared3D(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    return primitive_state[registered_variables.magnetic_field_index.x]**2 + \
           primitive_state[registered_variables.magnetic_field_index.y]**2 + \
           primitive_state[registered_variables.magnetic_field_index.z]**2


# @partial(jax.jit, static_argnames=["config", "registered_variables"])
# def conserved_state_from_primitive_mhd(
#     primitive_state: STATE_TYPE,
#     gamma: Union[float, Float[Array, ""]],
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
# ) -> STATE_TYPE:
#     """
#     For now only 3D.
#     """

#     rho = primitive_state[registered_variables.density_index]

#     u_squared = _u_squared3D(primitive_state, registered_variables)

#     p = primitive_state[registered_variables.pressure_index] # thermal pressure

#     b_squared = _b_squared3D(primitive_state, registered_variables)
    
#     # calculate total energy
#     E = total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma)

#     # create conserved state
#     conserved_state = primitive_state.at[registered_variables.pressure_index].set(E)

#     # set momentum density
#     conserved_state = conserved_state.at[registered_variables.velocity_index.x:registered_variables.velocity_index.z+1].set(
#         rho * primitive_state[registered_variables.velocity_index.x:registered_variables.velocity_index.z+1]
#     )

#     return conserved_state

# @partial(jax.jit, static_argnames=["config", "registered_variables"])
# def primitive_state_from_conserved_mhd(
#     conserved_state: STATE_TYPE,
#     gamma: Union[float, Float[Array, ""]],
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
# ) -> STATE_TYPE:
#     """
#     For now only 3D.
#     """

#     rho = conserved_state[registered_variables.density_index]
#     E = conserved_state[registered_variables.pressure_index]

#     ux = conserved_state[registered_variables.velocity_index.x] / rho
#     uy = conserved_state[registered_variables.velocity_index.y] / rho
#     uz = conserved_state[registered_variables.velocity_index.z] / rho

#     u_squared = ux**2 + uy**2 + uz**2

#     b_squared = _b_squared3D(conserved_state, registered_variables)

#     p = thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)

#     # set the primitive state

#     # pressure
#     primitive_state = conserved_state.at[registered_variables.pressure_index].set(p)

#     # velocities
#     primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(ux)
#     primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(uy)
#     primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(uz)

#     return primitive_state

# @partial(jax.jit, static_argnames=["registered_variables"])
# def total_pressure_from_conserved_mhd(
#     conserved_state: STATE_TYPE,
#     gamma: Union[float, Float[Array, ""]],
#     registered_variables: RegisteredVariables,
# ) -> FIELD_TYPE:
#     """
#     For now only 3D.
#     """

#     rho = conserved_state[registered_variables.density_index]
#     E = conserved_state[registered_variables.pressure_index]

#     ux = conserved_state[registered_variables.velocity_index.x] / rho
#     uy = conserved_state[registered_variables.velocity_index.y] / rho
#     uz = conserved_state[registered_variables.velocity_index.z] / rho

#     u_squared = ux**2 + uy**2 + uz**2

#     b_squared = _b_squared3D(conserved_state, registered_variables)

#     p_thermal = thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)

#     return p_thermal + 0.5 * b_squared

# ===========================================

# =============== Fluid physics ===============


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def get_absolute_velocity(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> Union[
    Float[Array, "num_cells"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"],
]:
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
        return jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
        )
    elif config.dimensionality == 3:
        return jnp.sqrt(
            primitive_state[registered_variables.velocity_index.x] ** 2
            + primitive_state[registered_variables.velocity_index.y] ** 2
            + primitive_state[registered_variables.velocity_index.z] ** 2
            + 1e-20
        )
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
    """Calculate the internal energy from the total energy.

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
    """Calculate the pressure from the total energy.

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

# @jax.jit
# def thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma):
#     """Calculate the pressure from the total energy in MHD.

#     Args:
#         E: The total energy.
#         rho: The density.
#         u_squared: The squared velocity.
#         b_squared: The squared magnetic field.
#         gamma: The adiabatic index.
#     Returns:
#         The pressure.
#     """
#     return (gamma - 1) * (E - 0.5 * rho * u_squared - 0.5 * b_squared)


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
def total_energy_from_primitives_mhd(
    rho, u_squared, p, b_squared, gamma
):
    return p / (gamma - 1) + 0.5 * rho * u_squared + 0.5 * b_squared


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
