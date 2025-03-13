# general
import jax
import jax.numpy as jnp
from functools import partial

# jf1uids constants
from jf1uids.option_classes.simulation_config import SPHERICAL

# jf1uids functions
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import total_energy_from_primitives_with_crs
from jf1uids.shock_finder.shock_finder import find_shock_zone, shock_sensor





@partial(jax.jit, static_argnames=['registered_variables', 'config'])
def inject_crs_at_strongest_shock_radial1D(primitive_state, gamma, helper_data, cosmic_ray_params, config, registered_variables, dt):
    """
    Compute the dissipated energy flux due to the strongest shock in the domain.

    https://arxiv.org/abs/1604.07399, section 3.1.2

    """

    # see also
    # https://github.com/LudwigBoess/DiffusiveShockAccelerationModels.jl/tree/main/src/mach_models
    # for models regarding the efficiency of the injection

    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency

    gamma_cr = 4/3
    gamma_gas = gamma

    num_cells = primitive_state.shape[1]
    indices = jnp.arange(num_cells)

    sensors = shock_sensor(primitive_state[registered_variables.pressure_index])

    velocity = primitive_state[registered_variables.velocity_index]

    div_v = jnp.zeros_like(velocity)
    div_v = div_v.at[1:-1].set((velocity[2:] - velocity[:-2]) / 2)

    sensors = jnp.where(div_v < 0, sensors, 0)

    max_shock_idx = jnp.argmax(sensors)

    left_idx, right_idx = find_shock_zone(primitive_state[registered_variables.pressure_index], primitive_state[registered_variables.velocity_index])

    left_idx = left_idx + 2

    # we only consider a shock moving from left to right
    # pre-shock is upstream in the shock frame
    # post-shock is downstream in the shock frame
    pre_shock_idx = right_idx + 1
    post_shock_idx = left_idx - 1

    shock_zone_mask = (indices >= left_idx) & (indices <= right_idx)

    shock_zone_size = jnp.sum(shock_zone_mask)

    # jax.debug.print("shock_zone_size: {sz}", sz = shock_zone_size)


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

    # jax.debug.print("gamma_eff1: {gamma_eff1}, gamma_eff2: {gamma_eff2}", gamma_eff1 = gamma_eff1, gamma_eff2 = gamma_eff2)

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

    # jax.debug.print("gamma_t: {gamma_t}", gamma_t = gamma_t)

    gamma1 = P1/e1 + 1
    gamma2 = P2/e2 + 1
    C = ((gamma2 + 1) * gamma_t + gamma2 - 1) * (gamma1 - 1)

                        # or 2 (?) like in Shock-accelerated cosmic rays and streaming instability in the adaptive mesh refinement code Ramses
    # M_1_sq = 1/gamma_eff1 * (gamma_t - 1) * C / (C - (gamma1 + 1 + (gamma1 - 1) * gamma_t) * (gamma2 - 1))

    M_1_sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))

    # jax.debug.print("M1: {M1}", M1 = jnp.sqrt(M_1_sq))
    


    # print rho2 and rho1
    # jax.debug.print("rho2: {rho2}, rho1: {rho1}", rho2 = rho2, rho1 = rho1)
    
    # only inject if the left index is positive, if left and right index are less than 20 apart and P2 > P1
    # criterion = (left_idx > 0) & (right_idx - left_idx < 20) & (P2 > P1) & (M_1_sq > M1_crit ** 2)
    # injection_efficiency = jax.lax.cond(criterion, lambda _: 0.0, lambda _: injection_efficiency, None)

    f_diss = e_diss * jnp.sqrt(M_1_sq) * c1 / x_s

    # calculate the shock surface
    if config.geometry == SPHERICAL:
        shock_radius = helper_data.geometric_centers[max_shock_idx]
        shock_surface = 4 * jnp.pi * shock_radius ** 2
    else:
        shock_surface = config.grid_spacing ** (config.dimensionality - 1)

    # jax.debug.print("shock_zone_size {sz}", sz = shock_zone_size)

    DeltaE_CR = f_diss * shock_surface * dt * injection_efficiency

    # TODO: finish, think about coupling to the energy equation,
    # so also remove from kinetic energy, adapt velocity accordingly

    # leads to crash in the beginning 
    # total energy is still an energy density, so we need to multiply by the volume to get the total energy
    # E_tot = total_energy_from_primitives_with_crs(primitive_state, registered_variables) * helper_data.cell_volumes

    # P = primitive_state[registered_variables.pressure_index]
    # PCR = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    # PGAS = P - PCR
    # E_tot = (PGAS / (gamma_gas - 1) + PCR / (gamma_cr - 1)) * helper_data.cell_volumes

    # deltaEtot = jnp.sum(jnp.where(shock_zone_mask, E_tot - E_tot[pre_shock_idx], 0))

    # deltaE_CR_shock_zone = jnp.where(shock_zone_mask, DeltaE_CR * (E_tot - E_tot[pre_shock_idx]) / deltaEtot, 0) / helper_data.cell_volumes

    # or just this
    # deltaE_CR_shock_zone = jnp.where(shock_zone_mask, DeltaE_CR, 0) / helper_data.cell_volumes / shock_zone_size

    # p_cr = primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    # p_cr_new = p_cr + deltaE_CR_shock_zone * (gamma_cr - 1)
    # n_cr_new = p_cr_new ** (1/gamma_cr)

    # primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].set(n_cr_new)

    injection_index = left_idx + 1
    p_cr_injection = primitive_state[registered_variables.cosmic_ray_n_index, injection_index] ** gamma_cr
    p_cr_injection_new = p_cr_injection + DeltaE_CR / helper_data.cell_volumes[injection_index] * (gamma_cr - 1)
    n_cr_injection_new = p_cr_injection_new ** (1/gamma_cr)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index, injection_index].set(n_cr_injection_new)

    # because we model the total pressure, increasing the cosmic ray pressure automatically
    # takes away gas pressure, if we want to take the energy from the kinetic energy, 
    # we would have to increase the total pressure and adapt the velocity to 
    # conserve energy

    # # also add to the total pressure
    # delta_p_shock = deltaE_CR_shock_zone * (gamma_cr - 1)
    # primitive_state = primitive_state.at[registered_variables.pressure_index].add(delta_p_shock)

    # # and update the velocity to conserve energy
    # v = primitive_state[registered_variables.velocity_index]
    # rho = primitive_state[registered_variables.density_index]
    # v_new = jnp.sqrt((0.5 * rho * v ** 2 - deltaE_CR_shock_zone) / (0.5 * rho)) # problem of negative values
    # primitive_state = primitive_state.at[registered_variables.velocity_index].set(v_new)

    # TODO: shock sharpening
    # ratio = P2_CRs / P2

    # mini_mask = (indices >= right_idx - 2) & (indices <= right_idx)
    # primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].set(jnp.where(mini_mask, (primitive_state[registered_variables.pressure_index] * ratio) ** (1/gamma_cr), primitive_state[registered_variables.cosmic_ray_n_index]))


    return primitive_state