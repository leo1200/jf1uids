# numerics
import jax
import jax.numpy as jnp

# plotting
from astropy.visualization import quantity_support
quantity_support()
import matplotlib.pyplot as plt

from unit_helpers import CodeUnits
from astropy import units as u
import astropy.constants as c
from weaver import Weaver
from astropy.constants import m_p

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

# =================== HELPER FUNCTIONS ====================

def calculate_pressure_from_internal_energy(e, rho, gamma):
  """
  Calculate pressure from internal energy.

    Parameters
    ----------
    e : float
      Internal energy per unit volume.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
      Pressure
  """
  return (gamma - 1) * e * rho

def calculate_internal_energy_from_energy(E, rho, vel):
    """
    Calculate internal energy from total energy.
    
        Parameters
        ----------
        E : float
        Total energy per unit volume.
        rho : float
        Density.
        u : float
        Velocity.
    
        Returns
        -------
        float
        Internal energy per unit volume.
    """
    return E / rho - 0.5 * vel**2

def calculate_total_energy_from_primitive_vars(rho, u, p, gamma):
    """
    Calculate total energy from primitive variables.
    
        Parameters
        ----------
        rho : float
        Density.
        u : float
        Velocity.
        p : float
        Pressure.
    
        Returns
        -------
        float
        Total energy per unit volume.
    """
    return p / (gamma - 1) + 0.5 * rho * u**2

def calculate_speed_of_sound(rho, p, gamma):
    """
    Calculate speed of sound.
    
        Parameters
        ----------
        rho : float
        Density.
        p : float
        Pressure.
        gamma : float
        Adiabatic index.
    
        Returns
        -------
        float
        Speed of sound.
    """
    return jnp.sqrt(gamma * p / rho)

def calculate_spherical_cell_volumes(r):
    """
    Calculate spherical cell volumes.
    
        Parameters
        ----------
        r : tensor
        Radial coordinate.
    
        Returns
        -------
        tensor
        Spherical cell volumes.
    """
    return 4/3 * jnp.pi * (r[1:]**3 - r[:-1]**3)

def add_mass(state, cell_volumes, min_index, max_index, mass):
    drho = mass / cell_volumes[min_index:max_index] / (max_index - min_index)
    state = state.at[0, min_index:max_index].add(drho)
    return state

def add_energy(state, cell_volumes, min_index, max_index, energy):
    # state[2] is the total energy per unit volume
    dE = energy / cell_volumes[min_index:max_index] / (max_index - min_index)
    state = state.at[2, min_index:max_index].add(dE)
    return state


# LIMITERS, TODO: CHECK IMPLEMENTATIONS
def minmod(a, b):
    # a and b are tensors of the same shape
    # minmod(a, b) = 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b))
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))

def maxmod(a, b):
    # maxmod(a, b) = 0.5 * (sign(a) + sign(b)) * max(abs(a), abs(b))
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.maximum(jnp.abs(a), jnp.abs(b))

def superbee(a, b):
    # superbee(a, b) = minmod(maxmod(a,b), minmod(2a, 2b)) where a * b > 0, else 0
    res = minmod(maxmod(a, b), minmod(2 * a, 2 * b))
    res = res.at[a * b <= 0].set(0)
    return res

# =========================================================

# ===================== GODUNOV SCHEME =====================

def get_initial_state(rho, u, p, gamma):
    """
    Get initial state.
    
        Parameters
        ----------
        rho : float
        Density.
        u : float
        Velocity.
        p : float
        Pressure.
    
        Returns
        -------
        Initial state tensor (rho, m = rho * u, E).
    """
    m = rho * u
    E = calculate_total_energy_from_primitive_vars(rho, u, p, gamma)
    return jnp.stack([rho, m, E], axis=0)

def get_center_flux(state, gamma):
    """
    Get flux tensor.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
    
        Returns
        -------
        Flux tensor (rho * u, rho * u^2 + p, u * (E + p)).
    """
    rho = state[0]
    m = state[1]
    E = state[2]
    u = m / rho
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)

def get_cfl_time_step(states, gamma, dr, C_CFL, dt_max):

    # TODO: combine with flux calculation, otherwise double computation

    rho = states[0]
    u = states[1] / states[0]
    E = calculate_total_energy_from_primitive_vars(states[0], states[1], states[2], gamma)
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    sound_speeds = calculate_speed_of_sound(states[0], p, gamma)

    # get the left and right states and fluxes
    states_left = states[:, :-1]
    states_right = states[:, 1:]

    # left and right sound speeds
    sound_speeds_left = sound_speeds[:-1]
    sound_speeds_right = sound_speeds[1:]

    # left and right velocities
    u_left = states_left[1] / states_left[0]
    u_right = states_right[1] / states_right[0]

    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_left + sound_speeds_left, u_right + sound_speeds_right), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_left - sound_speeds_left, u_right - sound_speeds_right), 0)

    # print left and right wave speeds
    # print("Left wave speeds: ", wave_speeds_left_minus)
    # print("Right wave speeds: ", wave_speeds_right_plus)

    # get the maximum wave speed
    max_wave_speed = jnp.maximum(jnp.max(jnp.abs(wave_speeds_right_plus)), jnp.max(jnp.abs(wave_speeds_left_minus)))

    # print("Wave speed: ", max_wave_speed)

    # calculate the time step
    dt = C_CFL * dr / max_wave_speed

    return jnp.minimum(dt, dt_max)


def get_interface_fluxes(states, fluxes, gamma):
    """
    Get the HLL fluxes at the interfaces.

    Parameters
    ----------
    states : tensor
        State tensor (rho, m = rho * u, E).
    fluxes : tensor
        Flux tensor (rho * u, rho * u^2 + p, u * (E + p)).
    gamma : float
    
    """

    rho = states[0]
    u = states[1] / states[0]

    # calculate the sound speeds
    E = calculate_total_energy_from_primitive_vars(states[0], states[1], states[2], gamma)
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    sound_speeds = calculate_speed_of_sound(states[0], p, gamma)

    # get the left and right states and fluxes
    # TODO: change in MUSCL-Hancock
    states_left = states[:, :-1]
    states_right = states[:, 1:]
    fluxes_left = fluxes[:, :-1]
    fluxes_right = fluxes[:, 1:]

    # left and right sound speeds
    sound_speeds_left = sound_speeds[:-1]
    sound_speeds_right = sound_speeds[1:]

    # left and right velocities
    u_left = states_left[1] / states_left[0]
    u_right = states_right[1] / states_right[0]

    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_left + sound_speeds_left, u_right + sound_speeds_right), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_left - sound_speeds_left, u_right - sound_speeds_right), 0)

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (wave_speeds_right_plus * fluxes_left - wave_speeds_left_minus * fluxes_right + wave_speeds_left_minus * wave_speeds_right_plus * (states_right - states_left)) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes

def geometric_source_term(state, r, alpha):
    """
    Geometric source term.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
        r : tensor
        Radial coordinate.
        alpha : float
        Geometry parameter.
    
        Returns
        -------
        Source term tensor.
    """
    rho = state[0]
    m = state[1]
    E = state[2]
    u = m / rho
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    # S = -alpha / r * [rho u, rho u^2, u (E + p)]
    return -alpha / r * jnp.stack([rho * u, rho * u**2, u * (E + p)], axis=0)


def godunov_update(state, dt, dx, gamma, r, alpha):
    """
    Godunov update.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
        dt : float
        Time step.
        dx : float
        Grid spacing.
    
        Returns
        -------
        Updated state tensor (rho, m = rho * u, E).
    """
    # get the fluxes at the cell centers
    fluxes = get_center_flux(state, gamma)

    # calculate the geometric source term
    source_term = geometric_source_term(state, r, alpha)

    # 2nd order source term handling
    state = state.at[:, 1:-1].add(dt / 2 * source_term[:, 1:-1])
    state = wind(state, r, dt / 2)

    # get the fluxes at the interfaces
    interface_fluxes = get_interface_fluxes(state, fluxes, gamma)

    state = state.at[:, 1:-1].add(-dt / dx * (interface_fluxes[:, 1:] - interface_fluxes[:, :-1]))
    # maybe to 2nd order source term split instead of 1st order

    # calculate the geometric source term
    source_term = geometric_source_term(state, r, alpha)
    state = state.at[:, 1:-1].add(dt / 2 * source_term[:, 1:-1])

    state = wind(state, r, dt / 2)

    # update the state

    # handle boundary conditions

    return state

def time_stepper(state, C_CFL, dx, gamma, dt_max, t_end, r, alpha):
    """
    Time stepper.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
        dt : float
        Time step.
        dx : float
        Grid spacing.
        t_end : float
        End time.
    
        Returns
        -------
        Updated state tensor (rho, m = rho * u, E).
    """
    t = 0

    progress = 0


    while t < t_end:

        # get the time step
        dt = get_cfl_time_step(state, gamma, dx, C_CFL, dt_max)

        # print(dt)

        # inner reflective boundary condition
        state = state.at[:, 0].set(state[:, 1])
        state = state.at[1, 0].set(-state[1, 1])

        # outer outflow boundary condition
        state = state.at[:, -1].set(state[:, -2])

        # update the state
        state = godunov_update(state, dt, dx, gamma, r, alpha)

        # increment the time
        t += dt

        # if dt passess another percent, print progress
        progress_new = int(t / t_end * 100)
        if progress_new > progress:
            print(f"Progress: {progress_new}%")
            progress = progress_new
    
    return state

# =========================================================

# ================== Physics Modules ======================

def dummy_wind(state, A, B, num_cells):
    state = state.at[0, 1:num_cells].add(A)
    state = state.at[2, 1:num_cells].add(B)
    return state

def wind(state, r, dt):

    # code units
    code_length = 3 * u.parsec
    code_mass = 1e5 * u.M_sun
    code_velocity = 1 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)


    # mass of the star
    M_star = 40 * u.M_sun

    num_cells = 10

    # wind features
    StellarWindFinalVelocity = 2000 * u.km / u.s
    StellarWindMassLossRate = 2.965e-3 / (1e6 * u.yr) * M_star

    m_dot = StellarWindMassLossRate.to(code_units.code_mass / code_units.code_time).value


    # mass injection
    r_inj = r[num_cells]
    V = 4/3 * jnp.pi * r_inj**3
    rho_dot = m_dot / V * dt
    state = state.at[0, 1:num_cells].add(rho_dot)

    # energy injection
    E_dot = 0.5 * StellarWindFinalVelocity**2 * StellarWindMassLossRate
    E_dot = E_dot.to(code_units.code_mass * code_units.code_velocity**2 / code_units.code_time).value / V * dt

    state = state.at[2, 1:num_cells].add(E_dot)

    # # energy injection
    # momentum_dot = StellarWindFinalVelocity.to(code_units.code_velocity) * state[0, 1:num_cells] * dt

    # # inject momentum
    # state = state.at[1, 1:num_cells].add(momentum_dot)


    return state


# =========================================================

# =============== EXAMPLE INITIAL CONDITIONS ===============

# for testing purposes, setup sod shock tube problem
N_grid = 400
L = 1.0
alpha = 2 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
shock_pos = 0.4
gamma = 5/3

dx = L / N_grid

# code units
code_length = 3 * u.parsec
code_mass = 1e5 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

r = jnp.linspace(0 + dx/2, L + dx/2, N_grid) * code_units.code_length

# ISM density
rho_0 = 2 * c.m_p / u.cm**3
# ISM pressure
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

print(rho_0.to(code_units.code_density))

# wanted final time
t_final = 1.5 * 1e4 * u.yr
code_units.print_simulation_parameters(t_final)

rho_init = jnp.ones(N_grid) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(N_grid)
p_init = jnp.ones(N_grid) * p_0.to(code_units.code_pressure).value

# get initial state
state = get_initial_state(rho_init, u_init, p_init, gamma)

# simulation settings
C_CFL = 0.4

t_end = t_final.to(code_units.code_time).value
dt_max = 0.005 * t_end

# print end time
print(f"End time: {t_end}")

# ========== WIND ===========


# ===========================

# run the simulation
state = time_stepper(state, C_CFL, dx, gamma, dt_max, t_end, r.to(code_units.code_length).value, alpha)

# =================================================================

# =========================== PLotting ============================

# mass of the star
M_star = 40 * u.M_sun

num_cells = 10

# wind features
StellarWindFinalVelocity = 2000 * u.km / u.s
StellarWindMassLossRate = 2.965e-3 / (1e6 * u.yr) * M_star

v_wind = StellarWindFinalVelocity
M_dot = StellarWindMassLossRate
rho_0 = 2 * m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

# get weaver solution
weaver = Weaver(v_wind, M_dot, rho_0, p_0)

current_time = t_final

# density
r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_density_weaver = r_density_weaver.to(u.parsec)
density_weaver = (density_weaver / m_p).to(u.cm**-3)

# velocity
r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_velocity_weaver = r_velocity_weaver.to(u.parsec)
velocity_weaver = velocity_weaver.to(u.km / u.s)

# pressure
r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_pressure_weaver = r_pressure_weaver.to(u.parsec)
pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)


# plot final state, density, velocity, pressure, internal energy
rho = state[0, 1:] * code_units.code_density
vel = state[1, 1:] / state[0, 1:] * code_units.code_velocity
E = state[2, 1:] * code_units.code_pressure
internal_energy = calculate_internal_energy_from_energy(E, rho, vel)
p = calculate_pressure_from_internal_energy(internal_energy, rho, gamma)
entropic_function = p / state[0, 1:]**gamma

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

r = r[1:]

axs[0].set_yscale("log")
axs[0].plot(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="radial 1d simulation")
axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")
axs[0].set_title("Density")
# xlim 0 to 0.3
# show legend
axs[0].legend()

axs[1].set_yscale("log")
axs[1].plot(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="radial 1d simulation")
axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")
axs[1].set_title("Pressure")
# xlim 0 to 0.3
# show legend
axs[1].legend()

axs[2].set_yscale("log")
axs[2].plot(r.to(u.parsec), vel.to(u.km / u.s), label="radial 1d simulation")
axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
axs[2].set_title("Velocity")
# ylim 1 to 1e4 km/s
axs[2].set_ylim(1, 1e4)
# show legend
axs[2].legend()

# #axs[1, 1].plot(r, internal_energy / rho)
# # use log scale
# axs[1, 1].set_yscale("log")
# axs[1, 1].plot(r.to(u.parsec), internal_energy)
# axs[1, 1].set_title("internal energy")
# # xlim 0 to 0.3

# tight layout
plt.tight_layout()

plt.savefig("godunov.png")

# =========================================================

