from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import total_energy_from_primitives_with_crs
from jf1uids.shock_finder.shock_finder import find_shock_zone, shock_sensor


import jax
import jax.numpy as jnp


from functools import partial


@partial(jax.jit, static_argnames=['registered_variables'])
def inject_crs_at_strongest_shock_radial1D(primitive_state, gamma, helper_data, cosmic_ray_params, registered_variables, dt):
    """
    Compute the dissipated energy flux due to the strongest shock in the domain.

    https://arxiv.org/abs/1604.07399, section 3.1.2

    """

    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency

    gamma_cr = 4/3
    gamma_gas = gamma

    num_cells = primitive_state.shape[1]
    indices = jnp.arange(num_cells)

    sensors = shock_sensor(primitive_state[registered_variables.pressure_index])
    max_shock_idx = jnp.argmax(sensors)

    left_idx, right_idx = find_shock_zone(primitive_state[registered_variables.pressure_index])
    shock_zone_mask = (indices >= left_idx) & (indices <= right_idx)

    shock_zone_size = right_idx - left_idx

    # jax.debug.print("left: {li}, right: {ri}", li = left_idx, ri = right_idx)

    # we only consider a shock moving from left to right
    # pre-shock is upstream in the shock frame
    # post-shock is downstream in the shock frame
    pre_shock_idx = right_idx + 1
    post_shock_idx = left_idx - 1

    # get the pre and post shock states
    rho1 = primitive_state[registered_variables.density_index, pre_shock_idx]
    P1 = primitive_state[registered_variables.pressure_index, pre_shock_idx]
    P1_CRs = primitive_state[registered_variables.cosmic_ray_n_index, pre_shock_idx] ** gamma_cr
    P1_gas = P1 - P1_CRs
    e1_gas = P1_gas / (gamma_gas - 1)
    e1_crs = P1_CRs / (gamma_cr - 1)
    e1 = e1_gas + e1_crs

    rho2 = primitive_state[registered_variables.density_index, post_shock_idx]
    P2 = primitive_state[registered_variables.pressure_index, post_shock_idx]
    P2_CRs = primitive_state[registered_variables.cosmic_ray_n_index, post_shock_idx] ** gamma_cr
    P2_gas = P2 - P2_CRs
    e2_gas = P2_gas / (gamma_gas - 1)
    e2_crs = P2_CRs / (gamma_cr - 1)
    e2 = e2_gas + e2_crs

    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma_eff2 = (gamma_cr * P2_CRs + gamma_gas * P2_gas) / P2

    c1 = jnp.sqrt(gamma_eff1 * P1 / rho1)

    e_th1 = P1_gas / ((gamma_gas - 1))
    e_th2 = P2_gas / ((gamma_gas - 1))

    e_cr1 = P1_CRs / (gamma_cr - 1)
    e_cr2 = P2_CRs / (gamma_cr - 1)

    x_s = rho2 / rho1

    e_diss = e_th2 - e_th1 * x_s ** gamma_gas + e_cr2 - e_cr1 * x_s ** gamma_cr

    # jax.debug.print("blbl")

    # M_1_sq = (P2 / P1 - 1) * x_s / (gamma_eff * (x_s - 1))

    gamma_t = P2 / P1
    gamma1 = P1/e1 + 1
    gamma2 = P2/e2 + 1
    C = ((gamma2 + 1) * gamma_t + gamma2 - 1) * (gamma1 - 1)

    M_1_sq = 1/gamma_eff2 * (gamma_t - 1) * C / (C - (gamma1 + 1 + (gamma1 - 1) * gamma_t) * (gamma2 -1))


    # print rho2 and rho1
    # jax.debug.print("rho2: {rho2}, rho1: {rho1}", rho2 = rho2, rho1 = rho1)
    
    # only inject if the left index is positive, if left and right index are less than 20 apart and P2 > P1
    # criterion = (left_idx > 0) & (right_idx - left_idx < 20) & (P2 > P1) & (M_1_sq > M1_crit ** 2)
    # injection_efficiency = jax.lax.cond(criterion, lambda _: 0.0, lambda _: injection_efficiency, None)

    f_diss = e_diss * jnp.sqrt(M_1_sq) * c1 / x_s

    # calculate the shock surface
    shock_radius = helper_data.geometric_centers[max_shock_idx]
    shock_surface = 4 * jnp.pi * shock_radius ** 2

    DeltaE_CR = f_diss * shock_surface * dt * injection_efficiency

    # TODO: finish, think about coupling to the energy equation,
    # so also remove from kinetic energy, adapt velocity accordingly

    # total energy is still an energy density, so we need to multiply by the volume to get the total energy
    E_tot = total_energy_from_primitives_with_crs(primitive_state, registered_variables) * helper_data.cell_volumes

    deltaEtot = jnp.sum(jnp.where(shock_zone_mask, E_tot - E_tot[pre_shock_idx], 0))

    deltaE_CR_shock_zone = jnp.where(shock_zone_mask, DeltaE_CR * (E_tot - E_tot[pre_shock_idx]) / deltaEtot, 0) / jnp.sum(jnp.where(shock_zone_mask, helper_data.cell_volumes, 0))
    delta_n_CR_shock_zone = (deltaE_CR_shock_zone * (gamma_cr - 1))  ** (1/gamma_cr)

    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].add(delta_n_CR_shock_zone)

    # this would be how to take the CR energy from the kinetic energy
    # otherwise we get problems at the simulation beginning
    # new_velocity = jnp.sqrt((0.5 * primitive_state[registered_variables.density_index] * primitive_state[registered_variables.velocity_index] ** 2 - deltaE_CR_shock_zone) / (0.5 * primitive_state[registered_variables.density_index]))
    # primitive_state = primitive_state.at[registered_variables.velocity_index].set(new_velocity)

    # here we take it from the thermal pool
    delta_p_gas_shock = deltaE_CR_shock_zone * (gamma - 1)
    primitive_state = primitive_state.at[registered_variables.pressure_index].add(-delta_p_gas_shock)

    return primitive_state