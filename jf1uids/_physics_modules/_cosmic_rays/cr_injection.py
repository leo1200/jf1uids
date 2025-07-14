# general
from typing import Union
import jax
import jax.numpy as jnp
from functools import partial

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union
from jf1uids.option_classes.simulation_config import STATE_TYPE

# jf1uids classes
from jf1uids._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig

# jf1uids constants
from jf1uids.option_classes.simulation_config import SPHERICAL

# jf1uids functions
from jf1uids.shock_finder.shock_finder import find_shock_zone

# NOTE: currently only supports 1d setups, TODO: generalize

@partial(jax.jit, static_argnames=['registered_variables', 'config'])
def inject_crs_at_strongest_shock(
    primitive_state: STATE_TYPE, 
    gamma: Union[float, Float[Array, ""]],
    helper_data: HelperData,
    cosmic_ray_params: CosmicRayParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]]
) -> STATE_TYPE:
    """
    Cosmic ray injection at shock fronts. 
    Currently only at the strongest shock in the domain.

    The implementation generally follows

    Pfrommer, Christoph, et al. "Simulating cosmic ray physics on a moving mesh." 
    Monthly Notices of the Royal Astronomical Society 465.4 (2017): 4500-4529.
    https://arxiv.org/abs/1604.07399

    and

    Dubois, Yohan, et al. "Shock-accelerated cosmic rays and streaming instability
    in the adaptive mesh refinement code Ramses."
    Astronomy & Astrophysics 631 (2019): A121.
    https://arxiv.org/abs/1907.04300

    Args:
        primitive_state: The primitive state array.
        gamma: The adiabatic index.
        helper_data: The helper data.
        cosmic_ray_params: The cosmic ray parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        dt: The time step.

    Returns:
        The primitive state array with injected cosmic rays.

    """

    num_cells = primitive_state.shape[1]

    # the injection efficiency is specified by the user
    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency
    # in future use e.g. models as implemented in
    # https://github.com/LudwigBoess/DiffusiveShockAccelerationModels.jl/tree/main/src/mach_models

    # currently for crs the adiabatic indices are hard coded
    gamma_cr = 4/3
    gamma_gas = gamma

    # find the strongest shock
    max_shock_idx, left_idx, right_idx = find_shock_zone(
        primitive_state,
        registered_variables,
    )

    # left_idx = max_shock_idx - 2
    # right_idx = max_shock_idx + 2

    # jax.debug.print("max shock index: {max_shock_idx}, left index: {left_idx}, right index: {right_idx}",
    #                  max_shock_idx=max_shock_idx,
    #                  left_idx=left_idx,
    #                  right_idx=right_idx)
    

    # def debug_shock_find(final_state, helper_data):
    #     import matplotlib.pyplot as plt
    #     from jf1uids.shock_finder.shock_finder import shock_sensor
    #     _, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    #     shock_index, shock_index_right, shock_index_left = find_shock_zone(final_state, registered_variables)

    #     gas_pressure = final_state[registered_variables.pressure_index] - final_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    #     cr_pressure = final_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    #     ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], final_state[registered_variables.pressure_index, shock_index - 15:shock_index + 15], "o--", color='black', label='total pressure')
    #     ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], gas_pressure[shock_index - 15:shock_index + 15], "o--", color='red', label='gas pressure')
    #     ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], cr_pressure[shock_index - 15:shock_index + 15], "o--", color='blue', label='cosmic ray pressure')

    #     ax1.axvline(helper_data.geometric_centers[shock_index], color='red', label='cell with max shock sensor')
    #     ax1.axvline(helper_data.geometric_centers[shock_index_right], color='blue', label='right shock boundary')
    #     ax1.axvline(helper_data.geometric_centers[shock_index_left], color='blue', label='left shock boundary')

    #     ax1.set_ylabel('Pressure')
    #     ax1.set_xlabel('Radius')
    #     ax1.legend(loc = 'upper right')

    #     ax2 = ax1.twinx()
    #     ax2.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], shock_sensor(final_state[registered_variables.pressure_index])[shock_index - 15:shock_index + 15], "o--", color='green')
    #     ax2.set_ylabel('Shock Sensor')

    #     if not jnp.any(jnp.isnan(final_state)):
    #         plt.savefig('shock_debug_{}.png'.format(shock_index))

    # jax.debug.callback(debug_shock_find, primitive_state, helper_data)


    # +2 leads to a smoother transition of the different pressure
    # components in the shock, but will lead to problems at lower
    # resolutions, also note the problem that CR pressure injected
    # in the broadened shock layer experiences PCR∇.u forces
    # (Dubois et al, 2019), so one might overinject effectively
    # left_idx = left_idx + 2

    # we only consider a shock moving from left to right
    # pre-shock is upstream in the shock frame
    # post-shock is downstream in the shock frame
    pre_shock_idx = right_idx + 1
    post_shock_idx = left_idx - 1

    # get the pre shock (upstream) state
    rho1 = primitive_state[registered_variables.density_index, pre_shock_idx] # density
    P1 = primitive_state[registered_variables.pressure_index, pre_shock_idx] # total pressure
    P1_CRs = primitive_state[registered_variables.cosmic_ray_n_index, pre_shock_idx] ** gamma_cr # cosmic ray pressure
    P1_gas = P1 - P1_CRs # gas pressure
    e1_gas = P1_gas / (gamma_gas - 1) # gas energy / volume
    e1_crs = P1_CRs / (gamma_cr - 1) # cosmic ray energy / volume
    e1 = e1_gas + e1_crs # total energy / volume

    # get the post shock state
    rho2 = primitive_state[registered_variables.density_index, post_shock_idx]
    P2 = primitive_state[registered_variables.pressure_index, post_shock_idx]
    P2_CRs = primitive_state[registered_variables.cosmic_ray_n_index, post_shock_idx] ** gamma_cr
    P2_gas = P2 - P2_CRs
    e2_gas = P2_gas / (gamma_gas - 1)
    e2_crs = P2_CRs / (gamma_cr - 1)
    e2 = e2_gas + e2_crs

    # get the effective adiabatic index
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1

    # calculate the pre-shock sound speed
    c1 = jnp.sqrt(gamma_eff1 * P1 / rho1)

    # density ratio
    x_s = rho2 / rho1

    # jax.debug.print("xs: {x_s}, gamma_eff1: {gamma_eff1}, c1: {c1}",
    #                  x_s=x_s,
    #                  gamma_eff1=gamma_eff1,
    #                  c1=c1)

    # pre-shock mach number
    M_1_sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))
    # M_1_sq = 100
    # alternatively one might use formula 16 in Dubois et al, 2019

    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma_eff2 = (gamma_cr * P2_CRs + gamma_gas * P2_gas) / P2

    gamma1 = P1 / e1 + 1
    gamma2 = P2 / e2 + 1

    gammat = P2/P1

    C = ((gamma2 + 1) * gammat + gamma2 - 1) * (gamma1 - 1)
    # M_1_sq = 1/gamma_eff2 * (gammat) * C / (C - ((gamma1 + 1) - (gamma1 - 1) * gammat) * (gamma2 - 1))

    # M_1_sq = 1 / (2 * gamma_eff1) * (gammat * (gamma_eff1 + 1) + gamma_eff1 - 1)

    # jax.debug.print("M1sq: {M1sq}, gamma_eff1: {gamma_eff1}, gammat: {gammat}, C: {C}",
    #                  M1sq=M_1_sq,
    #                  gamma_eff1=gamma_eff1,
    #                  gammat=gammat,
    #                  C=C)

    # M_1_sq = 10

    # dissipated energy density
    e_diss = e2_gas - e1_gas * x_s ** gamma_gas + e2_crs - e1_crs * x_s ** gamma_cr

    # dissipated flux
    f_diss = e_diss * jnp.sqrt(M_1_sq) * c1 / x_s

    # calculate the shock surface
    if config.geometry == SPHERICAL:
        shock_radius = helper_data.geometric_centers[max_shock_idx]
        shock_surface = 4 * jnp.pi * shock_radius ** 2
    else:
        shock_surface = config.grid_spacing ** (config.dimensionality - 1)

    # energy to be injected in the form of cosmic ray pressure
    DeltaE_CR = f_diss * shock_surface * dt * injection_efficiency

    # # get a mask for the shock, if an injection as done in Pfommer et al. 2017 is desired
    # indices = jnp.arange(num_cells)
    # shock_zone_mask = (indices >= left_idx) & (indices <= max_shock_idx)

    # cosmic_ray_pressure = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    # gas_pressure = primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    # e_th = gas_pressure / (gamma_gas - 1)
    # e_cr = cosmic_ray_pressure / (gamma_cr - 1)
    # E_tot = (e_th + e_cr) * helper_data.cell_volumes
    # DeltaEtot = jnp.sum(jnp.where(shock_zone_mask, E_tot - E_tot[right_idx], 0))
    # DeltaE_CR_split = DeltaE_CR * (E_tot - E_tot[right_idx]) / DeltaEtot
    # DeltaE_CR_split = jnp.where(shock_zone_mask, DeltaE_CR_split, 0)

    # # to be injected cosmic ray pressure
    # p_cr_injection = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    # # updated cosmic ray pressure
    # p_cr_injection_new = p_cr_injection + DeltaE_CR_split / helper_data.cell_volumes * (gamma_cr - 1)
    # # note that we work with n_cr = P_CR ^ (1 / gamma_cr) to describe the cosmic rays
    # n_cr_injection_new = p_cr_injection_new ** (1/gamma_cr)
    # primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].set(n_cr_injection_new)

    # # we want energy and not pressure conservation, so the total pressure must be adapted
    # # TODO: check energy conservation thoroughly
    # delta_p_gas = DeltaE_CR_split / helper_data.cell_volumes * (gamma_gas - 1)
    # p_gas_new = primitive_state[registered_variables.pressure_index] - p_cr_injection - delta_p_gas
    # total_pressure_new = p_gas_new + p_cr_injection_new

    # # update the total pressure
    # primitive_state = primitive_state.at[registered_variables.pressure_index].set(total_pressure_new)

    # inject into a single post-shock cell
    injection_index = left_idx # + 1

    # to be injected cosmic ray pressure
    p_cr_injection = primitive_state[registered_variables.cosmic_ray_n_index, injection_index] ** gamma_cr
    # updated cosmic ray pressure
    p_cr_injection_new = p_cr_injection + DeltaE_CR / helper_data.cell_volumes[injection_index] * (gamma_cr - 1)
    # note that we work with n_cr = P_CR ^ (1 / gamma_cr) to describe the cosmic rays
    n_cr_injection_new = p_cr_injection_new ** (1/gamma_cr)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index, injection_index].set(n_cr_injection_new)

    # we want energy and not pressure conservation, so the total pressure must be adapted
    # TODO: check energy conservation thoroughly
    delta_p_gas = DeltaE_CR / helper_data.cell_volumes[injection_index] * (gamma_gas - 1)
    p_gas_new = primitive_state[registered_variables.pressure_index, injection_index] - p_cr_injection - delta_p_gas
    total_pressure_new = p_gas_new + p_cr_injection_new

    # jax.debug.print("old total pressure: {old_pressure}, new total pressure: {new_pressure}",
    #                  old_pressure=primitive_state[registered_variables.pressure_index],
    #                  new_pressure=total_pressure_new)

    # update the total pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index, injection_index].set(total_pressure_new)

    return primitive_state