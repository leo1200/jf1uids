import numpy as np
import jax.numpy as jnp
import jax
from astropy.io import fits
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from functools import partial

# process table
def get_mass_entry(data,mass,rotation):
    """
        Returns the table for a given initil mass 
        considering if the rotating or non-rotatig model should be returned.
    """
    idx = np.where(data['Mini']==mass)
    if rotation:
        rot = np.where(data[idx]['Rot'] == 'r')
    else:
        rot = np.where(data[idx]['Rot'] == 'n')
    return data[idx][rot]
    
    
# function definition to split the model tracks into different evolutionary phases
# see Georgy+2012 (https://arxiv.org/pdf/1203.5243.pdf) for more details

def is_wr_type(teff,x):
    idx = (teff > 1e4) * (x < 0.3)
    return np.where(idx==True)

def is_o_type(teff,x):
    idx = (teff > (10**4.5)) * (x >= 0.3)
    return np.where(idx==True)

def is_wc_type(teff,x,c12,c13,n):
    idx = (teff > 1e4) * (x == 0.) * ((c12+c13) > n)
    return np.where(idx==True)
    
    
# wind velocities for different evolutionary phases
# see Gatto+2012 (https://arxiv.org/pdf/1606.05346.pdf) for more details

# linear interpolation 700 km/s at T_eff=2e4 K to 2800 km/s at T_eff=8e4 K 
def wc_wind(teff):
    m = (2800 - 700) / (8e4 - 2e4) #k
    x = teff - 2e4
    b = 700
    return m * x + b

# linear interpolation 700 km/s at T_eff=2e4 K to 2100 km/s at T_eff=5e4 K 
def wnl_wind(teff):
    m = (2100 - 700) / (5e4 - 2e4) #k
    x = teff - 2e4
    b = 700
    return m * x + b

def wind_pulse(teff,vesc):
    #For O-type stars
    low_teff = np.where(teff < 1.8e4)
    high_teff = np.where(teff > 2.3e4)
    interp = ((2.45 - 1.3) / (2.3e4 - 1.8e4)) * (teff - 1.84e4) + 1.3
    interp[low_teff] = 1.3
    interp[high_teff] = 2.45
    
    return interp * vesc

def supergiant_wind(L):
    return 10 * (L / 3e4)**0.25
    
# combine all functions above to return one complete model

def get_wind_velocity(teff,mass,L,mini,x,c12,c13,n):
    # L = 4*pi*R^2*sigma_boltz*T^4
    sigma_b = 5.67e-8 #
    Lsun = 3.8e26 #Watts
    r = np.sqrt(L*Lsun/(4*np.pi*sigma_b*teff**4)) #in meter
    
    # vesc = sqrt(2GM/r)
    G = 6.674e-11 # m^3 / kg^-1 s^-2
    vesc = np.sqrt(2*G*mass*2e30/r) / 1000 # in km/s
    
    #v_wind = np.zeros_like(time)
    #o = is_o_type(teff,x)
    v_wind = wind_pulse(teff,vesc)
    
    wr = is_wr_type(teff,x)
    v_wind[wr] = wnl_wind(teff)[wr]
    
    wc = is_wc_type(teff,x,c12,c13,n)
    v_wind[wc] = wc_wind(teff)[wc]

    return v_wind
  
  
  
def wind_parameters(data,mass,rotation):
    """
        Given some data table data this function returns the wind parameters wind velocity 
        and mass loss rate for a model of initial mass mass and for the rotating or non-rotating model.
        
        return values: 
            time in yr
            mass loss in log10
            wind velocity km/s
    """
    
    model = get_mass_entry(data,mass,rotation)
    wind = get_wind_velocity(10**model['logTe'],model['Mass'],10**model['logL'],model['Mini'],model['X'],model['C12'],model['C13'],model['N14'])

    return [model['Time'],model['logdM_dt'],wind]

###Numpy function to generate wind parameters for all particles
def get_wind_parameters(particle_masses):
    #### TODO: append function ersetzen
    N = len(particle_masses)
    hdul = fits.open('ekstroem+2012.fit')
    t_yr, mass_rates, vel_scales = [], [], []
    for idx in range(N):
        t, m, v = wind_parameters(hdul[1].data, particle_masses[idx], True)
        t_yr.append(t)
        mass_rates.append(m)
        vel_scales.append(v)

    t_yr = jnp.array(t_yr)
    log_mass_rates = jnp.array(mass_rates)
    vel_scales_kms = jnp.array(vel_scales) 
    return t_yr, log_mass_rates, vel_scales_kms


# @jaxtyped(typechecker=typechecker)
@jax.jit
def get_current_wind_params(mass_rates_value, vel_scales_value, current_time, time_value):
    interp_fn = lambda x_i, y_i: piecewise_linear(current_time, x_i, y_i)
    mass_rates = jax.vmap(interp_fn)(time_value, mass_rates_value)  
    vel_scales = jax.vmap(interp_fn)(time_value, vel_scales_value)    

    return mass_rates, vel_scales


# @partial(jax.jit, static_argnames=['x', 'y'])
def piecewise_linear(t, x, y):
    """
    Piecewise-linear interpolation of (x, y) at points t.

    Args:
      t: current time
      x: time coordinates from fits file
      y: f(x)

    Returns:
      Interpolated values at t
    """
    # find insertion indices (right side)
    idx = jnp.searchsorted(x, t, side="right")   # idx in [0, n]
    n = x.shape[0]
    # indices for interpolation segment (left/right)
    left_idx = jnp.clip(idx - 1, 0, n - 2)  # left in [0, n-2]
    right_idx = left_idx + 1

    x_left = x[left_idx]
    x_right = x[right_idx]
    y_left = y[left_idx]
    y_right = y[right_idx]

    denom = x_right - x_left
    frac = jnp.where(denom == 0, 0.0, (t - x_left) / denom)
    interp = y_left + frac * (y_right - y_left)

    # left extrapolation: use slope of first segment
    first_denom = x[1] - x[0]
    first_slope = jnp.where(first_denom == 0, 0.0, (y[1] - y[0]) / first_denom)
    left_extrap = y[0] + first_slope * (t - x[0])

    right_clamp = y[-1]

    # pick between left_extrap, interp, and right_clamp
    is_left = t < x[0]
    is_right = t > x[-1]
    out = jnp.where(is_left, left_extrap, interp)
    out = jnp.where(is_right, right_clamp, out)

    return out