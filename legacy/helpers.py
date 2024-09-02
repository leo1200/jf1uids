# numerics
import jax
import jax.numpy as jnp

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

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
  return (gamma - 1) * rho * e

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
    return E / rho - 0.5 * u**2

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
