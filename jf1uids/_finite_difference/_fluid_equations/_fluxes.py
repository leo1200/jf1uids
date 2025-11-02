import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float

from typing import Union

from jf1uids._finite_difference._fluid_equations._equations import total_energy_from_primitives_mhd, total_pressure_from_conserved_mhd
from jf1uids.variable_registry.registered_variables import AxisInfo, RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)

# We only define the flux in x-direction here,
# since the other directions can be obtained
# by permuting the arrays accordingly.
@partial(
    jax.jit, static_argnames=["config", "registered_variables"]
)
def _mhd_flux_x(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    
    # retrieve necessary quantities
    rho = primitive_state[registered_variables.density_index]
    v_x = primitive_state[registered_variables.velocity_index.x]
    v_y = primitive_state[registered_variables.velocity_index.y]
    v_z = primitive_state[registered_variables.velocity_index.z]
    B_x = primitive_state[registered_variables.magnetic_index.x]
    B_y = primitive_state[registered_variables.magnetic_index.y]
    B_z = primitive_state[registered_variables.magnetic_index.z]
    p_gas = primitive_state[registered_variables.pressure_index]

    # compute derived quantities
    b_squared = B_x**2 + B_y**2 + B_z**2
    v_squared = v_x**2 + v_y**2 + v_z**2
    total_pressure = p_gas + 0.5 * b_squared
    v_dot_B = v_x * B_x + v_y * B_y + v_z * B_z
    E = total_energy_from_primitives_mhd(
        rho,
        v_squared,
        p_gas,
        b_squared,
        gamma,
    )

    # compute fluxes
    flux = jnp.zeros_like(primitive_state)
    flux = flux.at[registered_variables.density_index].set(rho * v_x)
    flux = flux.at[registered_variables.velocity_index.x].set(rho * v_x**2 + total_pressure - B_x**2)
    flux = flux.at[registered_variables.velocity_index.y].set(rho * v_x * v_y - B_x * B_y)
    flux = flux.at[registered_variables.velocity_index.z].set(rho * v_x * v_z - B_x * B_z)
    flux = flux.at[registered_variables.pressure_index].set((E + total_pressure) * v_x - v_dot_B * B_x)
    flux = flux.at[registered_variables.magnetic_index.x].set(0.0)
    flux = flux.at[registered_variables.magnetic_index.y].set(B_y * v_x - B_x * v_y)
    flux = flux.at[registered_variables.magnetic_index.z].set(B_z * v_x - B_x * v_z)

    return flux


# # TODO: improve readability
# @partial(
#     jax.jit, static_argnames=["config", "registered_variables", "flux_axis"]
# )
# def _mhd_flux(
#     conserved_state: STATE_TYPE,
#     gamma: Union[float, Float[Array, ""]],
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
#     flux_axis: AxisInfo,
# ) -> STATE_TYPE:
#     """
#     Compute the MHD fluxes, for now only in 3D.

#     NOTE: We work on the conserved state here, not the primitive state!
#     (Because I'm currently writing this for the finite difference MHD solver, 
#     where we handle conserved variables.)

#     Conserved state:
#         [ ρ,

#           ρ v_x,
#           ρ v_y,
#           ρ v_z,

#           E,

#           B_x,
#           B_y,
#           B_z,

#         ]

#     The flux in x-direction is given by:

#         F = [
#             ρ v_x,                         // density_index

#             ρ v_x^2 + P_tot - B_x^2,       // velocity_index.x
#             ρ v_x v_y - B_x B_y,           // velocity_index.y
#             ρ v_x v_z - B_x B_z,           // velocity_index.z

#             (E + P_tot) v_x - (B · v) B_x  // energy_index

#             0,                             // magnetic_index.x
#             B_y v_x - B_x v_y,             // magnetic_index.y
#             B_z v_x - B_x v_z,             // magnetic_index.z

#         ]

#     The flux in y-direction is given by:
#         F = [
#             ρ v_y,                         // density_index

#             ρ v_y v_x - B_y B_x,           // velocity_index.x
#             ρ v_y^2 + P_tot - B_y^2,       // velocity_index.y
#             ρ v_y v_z - B_y B_z,           // velocity_index.z

#             (E + P_tot) v_y - (B · v) B_y  // energy_index

#             B_x v_y - B_y v_x,             // magnetic_index.x
#             0,                             // magnetic_index.y
#             B_z v_y - B_y v_z,             // magnetic_index.z

#         ]

#     The flux in z-direction is given by:
#         F = [
#             ρ v_z,                         // density_index

#             ρ v_z v_x - B_z B_x,           // velocity_index.x
#             ρ v_z v_y - B_z B_y,           // velocity_index.y
#             ρ v_z^2 + P_tot - B_z^2,       // velocity_index.z

#             (E + P_tot) v_z - (B · v) B_z  // energy_index

#             B_x v_z - B_z v_x,             // magnetic_index.x
#             B_y v_z - B_z v_y,             // magnetic_index.y
#             0,                             // magnetic_index.z
#         ]

#     where P_tot = p + 0.5 * |B|^2 is the total pressure.

#     """

#     total_pressure = total_pressure_from_conserved_mhd(
#         conserved_state, gamma, registered_variables
#     )

#     # we can get from the conserved state to the flux by
#     # - add the total pressure to the pressure_index
#     # - multipy everything with conserved_state[axis_info.velocity_index] / density
#     # - add the pressure term to the axis_info.velocity_index
#     # - subtract the the magnetic terms * axis_info.magnetic_index from the velocity components
#     # - subtract the velocity terms * axis_info.magnetic_index from the magnetic components
#     #  - subtract the (B · v) B_axis from the energy component
#     # - set the field at axis_info.magnetic_index to zero

#     flux_vector = conserved_state.at[registered_variables.pressure_index].add(
#         total_pressure
#     )

#     v_axis = conserved_state[flux_axis.velocity_index] / conserved_state[
#         registered_variables.density_index
#     ]

#     flux_vector = v_axis * flux_vector

#     flux_vector = flux_vector.at[flux_axis.velocity_index].add(
#         total_pressure
#     )

#     flux_vector = flux_vector.at[
#         registered_variables.velocity_index.x:registered_variables.velocity_index.z + 1
#     ].add(
#         -conserved_state[registered_variables.magnetic_index.x:registered_variables.magnetic_index.z + 1]
#         * conserved_state[flux_axis.magnetic_index]
#     )

#     flux_vector = flux_vector.at[
#         registered_variables.magnetic_index.x:registered_variables.magnetic_index.z + 1
#     ].add(
#         -conserved_state[
#             registered_variables.velocity_index.x:registered_variables.velocity_index.z + 1
#         ] / conserved_state[registered_variables.density_index]
#         * conserved_state[flux_axis.magnetic_index]
#     )

#     v_dot_B = (
#         conserved_state[registered_variables.magnetic_index.x]
#         * conserved_state[registered_variables.velocity_index.x]
#         + conserved_state[registered_variables.magnetic_index.y]
#         * conserved_state[registered_variables.velocity_index.y]
#         + conserved_state[registered_variables.magnetic_index.z]
#         * conserved_state[registered_variables.velocity_index.z]
#     ) / conserved_state[registered_variables.density_index]
#     flux_vector = flux_vector.at[registered_variables.pressure_index].add(
#         -v_dot_B * conserved_state[flux_axis.magnetic_index]
#     )

#     flux_vector = flux_vector.at[flux_axis.magnetic_index].set(0.0)

#     return flux_vector