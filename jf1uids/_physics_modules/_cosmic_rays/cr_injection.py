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

    # the injection efficiency is specified by the user
    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency
    # in future use e.g. models as implemented in
    # https://github.com/LudwigBoess/DiffusiveShockAccelerationModels.jl/tree/main/src/mach_models

    # currently for crs the adiabatic indices are hard coded
    gamma_cr = 4/3
    gamma_gas = gamma

    # find the strongest shock
    max_shock_idx, left_idx, right_idx = find_shock_zone(
        primitive_state[registered_variables.pressure_index],
        primitive_state[registered_variables.velocity_index]
    )

    # +2 leads to a smoother transition of the different pressure
    # components in the shock, but will lead to problems at lower
    # resolutions, also note the problem that CR pressure injected
    # in the broadened shock layer experiences PCR∇.u forces
    # (Dubois et al, 2019), so one might overinject effectively
    left_idx = left_idx + 2

    # we only consider a shock moving from left to right
    # pre-shock is upstream in the shock frame
    # post-shock is downstream in the shock frame
    pre_shock_idx = right_idx + 1
    post_shock_idx = left_idx - 1

    # get a mask for the shock, if an injection as done in Pfommer et al. 2017 is desired
    # for us it proved to be simpler and similar in results to just inject in one post-shock cell
    # indices = jnp.arange(num_cells)
    # shock_zone_mask = (indices >= left_idx) & (indices <= right_idx)
    # shock_zone_size = jnp.sum(shock_zone_mask)

    # get the pre shock (upstream) state
    rho1 = primitive_state[registered_variables.density_index, pre_shock_idx] # density
    P1 = primitive_state[registered_variables.pressure_index, pre_shock_idx] # total pressure
    P1_CRs = primitive_state[registered_variables.cosmic_ray_n_index, pre_shock_idx] ** gamma_cr # cosmic ray pressure
    P1_gas = P1 - P1_CRs # gas pressure
    e1_gas = P1_gas / (gamma_gas - 1) # gas energy / volume
    e1_crs = P1_CRs / (gamma_cr - 1) # cosmic ray energy / volume

    # get the post shock state
    rho2 = primitive_state[registered_variables.density_index, post_shock_idx]
    P2 = primitive_state[registered_variables.pressure_index, post_shock_idx]
    P2_CRs = primitive_state[registered_variables.cosmic_ray_n_index, post_shock_idx] ** gamma_cr
    P2_gas = P2 - P2_CRs
    e2_gas = P2_gas / (gamma_gas - 1)
    e2_crs = P2_CRs / (gamma_cr - 1)

    # get the effective adiabatic index
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1

    # calculate the pre-shock sound speed
    c1 = jnp.sqrt(gamma_eff1 * P1 / rho1)

    # density ratio
    x_s = rho2 / rho1

    # pre-shock mach number
    M_1_sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))
    # alternatively one might use formula 16 in Dubois et al, 2019

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

    # inject into a single post-shock cell
    injection_index = left_idx + 1

    # to be injected cosmic ray pressure
    p_cr_injection = primitive_state[registered_variables.cosmic_ray_n_index, injection_index] ** gamma_cr
    # updated cosmic ray pressure
    p_cr_injection_new = p_cr_injection + DeltaE_CR / helper_data.cell_volumes[injection_index] * (gamma_cr - 1)
    # note that we work with n_cr = P_CR ^ (1 / gamma_cr) to describe the cosmic rays
    n_cr_injection_new = p_cr_injection_new ** (1/gamma_cr)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index, injection_index].set(n_cr_injection_new)

    # because we model the total pressure, increasing the cosmic ray pressure automatically
    # takes away gas pressure, if we want to take the energy from the kinetic energy, 
    # we would have to increase the total pressure and adapt the velocity to 
    # conserve energy

    return primitive_state