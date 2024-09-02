# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

# =================== HELPER FUNCTIONS ====================

def calculate_pressure_from_internal_energy(e, gamma):
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
  return (gamma - 1) * e

def calculate_internal_energy_from_energy(E, rho, u):
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
    return E - 0.5 * rho * u**2

def calculate_total_energy_from_primitive_vars(rho, u, p):
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


# =========================================================

def get_initial_state(rho, u, p):
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
    E = calculate_total_energy_from_primitive_vars(rho, u, p)
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
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)

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

    # calculate the sound speeds
    E = calculate_total_energy_from_primitive_vars(states[0], states[1], states[2])
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), gamma)
    sound_speeds = calculate_speed_of_sound(states[0], p, gamma)

    # get the left and right states and fluxes
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

def godunov_update(state, dt, dx, gamma):
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

    # get the fluxes at the interfaces
    interface_fluxes = get_interface_fluxes(state, fluxes, gamma)

    # print the interface fluxes
    # print(interface_fluxes)

    # update the state
    state = state.at[:, 1:-1].set(state[:, 1:-1] - dt / dx * (interface_fluxes[:, 1:] - interface_fluxes[:, :-1]))

    # print(dt / dx * (interface_fluxes[:, 1:] - interface_fluxes[:, :-1]))

    # handle boundary conditions

    return state

def time_stepper(state, dt, dx, gamma, t_end):
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
    while t < t_end:

        # get the time step
        dt = dt # possibly adjust the time step

        # update the state
        state = godunov_update(state, dt, dx, gamma)

        # increment the time
        t += dt
    
    return state

# =============== EXAMPLE INITIAL CONDITIONS ===============

# for testing purposes, setup sod shock tube problem
N_grid = 100
L = 1.0
r = jnp.linspace(0, L, N_grid)
alpha = 0 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
shock_pos = 0.5
gamma = 5/3

# initial conditions
rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.zeros_like(r)
p = jnp.where(r < shock_pos, 1.0, 0.1)

# get initial state
state = get_initial_state(rho, u, p)

# simulation settings
dt = 0.0001
dx = L / N_grid
t_end = 0.2

# run the simulation
state = time_stepper(state, dt, dx, gamma, t_end)

# plot final state, density, velocity, pressure, internal energy
rho = state[0]
u = state[1] / state[0]
E = state[2]
internal_energy = calculate_internal_energy_from_energy(state[2], state[0], u)
p = calculate_pressure_from_internal_energy(internal_energy, gamma)
entropic_function = p / state[0]**gamma

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(r, rho)
axs[0, 0].set_title("Density")

axs[0, 1].plot(r, p)
axs[0, 1].set_title("Pressure")

axs[1, 0].plot(r, u)
axs[1, 0].set_title("Velocity")

axs[1, 1].plot(r, internal_energy / rho)
axs[1, 1].set_title("specific internal energy")

plt.savefig("sod_shock_tube.png")

# =========================================================

