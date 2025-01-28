import numpy as np
import jax.numpy as jnp

def create_turb_field(Ndim, A0, slope, kmin, kmax, seed = None):
    """Creates a turbulent field with given slope, amplitude
       and cutoffs in Fourier space for a uniform grid in 3D.

    Parameters
    ----------
    Ndim: int
        the number of grid points in each dimension
    A0: float
        the amplitude of the field
    slope: float
        the slope of the power spectrum
    kmin: float
        the minimum wavenumber
    kmax: float
        the maximum wavenumber
    
    Returns
    -------
    rfield: np.ndarray
        the real field
    """
    
    # construct the k-vectors
    # wave number bin centers in cycles per unit
    # of the sample spacing
    d = 1
    k = np.fft.fftfreq(Ndim, d=d)*(d*Ndim)
    # * (d * Ndim) gets the actual wavenumbers, so
    # k = [0, 1, 2, ..., Ndim/2-1, -Ndim/2, ..., -1] for Ndim even
    # k = [0, 1, 2, ..., Ndim/2, -Ndim/2+1, ..., -1] for Ndim odd
    kx, ky, kz = np.meshgrid(k, k, k)
    k3d = np.sqrt(kx**2 + ky**2 + kz**2)

    # create the amplitudes of the power spectrum A0 * k**slope, kmin < k < kmax
    ampli  = np.zeros((Ndim,Ndim,Ndim), dtype=np.float64)
    idx = np.where((k3d < kmin) | (k3d > kmax))
    ampli[idx] = 0.0
    idx = np.where((k3d >= kmin) & (k3d <= kmax))
    ampli[idx] = A0 * np.power(k3d[idx],slope)

    # create random phase for the field in Fourier space
    if seed is not None:
        np.random.seed(seed)
    
    phase = np.random.uniform(low=0.0, high=2.*np.pi, size=Ndim**3).reshape(Ndim, Ndim, Ndim)
    # construct the fourier field with the given amplitude and phase
    ffield = ampli*np.cos(phase) + ampli*np.sin(phase)*1j

    # for a real field v, v* = v in real space, so by
    # the definition of the Fourier transform
    # \hat{v}_k = \hat{v}_{-k}^*
    # we enforce this to have a real velocity field
    for i in range(Ndim):
        # print("outer loop iteration {} of {}".format(i, Ndim))
        for j in range(Ndim):
            for k in range(Ndim//2 + 1):
                ffield[i,j,k] = np.conjugate(ffield[-i,-j,-k])

    # also as ffield_[0,0,0] = ffield*_[-0,-0,-0], ffield_[0,0,0] must be real
    # likewise as of aliasing
    # ffield_[Ndim//2, Ndim//2, Ndim//2] = ffield*_[-Ndim//2, -Ndim//2, -Ndim//2] = ffield*_[Ndim//2, Ndim//2, Ndim//2]
    # by the same reasoning
    # ffield_[Ndim//2, Ndim//2, 0], ffiedl_[Ndim//2, 0, Ndim//2], ffield_[0, Ndim//2, Ndim//2] must be real
    # and
    # ffield_[Ndim//2, 0, 0], ffield_[0, Ndim//2, 0], ffield_[0, 0, Ndim//2] must be real
    # in these cases we just take the absolute value of the complex number
    ffield[Ndim//2, Ndim//2, Ndim//2] = np.abs(ffield[Ndim//2, Ndim//2, Ndim//2])
    ffield[Ndim//2, Ndim//2, 0] = np.abs(ffield[Ndim//2, Ndim//2, 0])
    ffield[Ndim//2, 0, Ndim//2] = np.abs(ffield[Ndim//2, 0, Ndim//2])
    ffield[0, Ndim//2, Ndim//2] = np.abs(ffield[0, Ndim//2, Ndim//2])
    ffield[Ndim//2, 0, 0] = np.abs(ffield[Ndim//2, 0, 0])
    ffield[0, Ndim//2, 0] = np.abs(ffield[0, Ndim//2, 0])
    ffield[0, 0, Ndim//2] = np.abs(ffield[0, 0, Ndim//2])
    ffield[0, 0, 0] = np.abs(ffield[0, 0, 0])

    # get the real field
    rfield = np.fft.ifftn(ffield)

    # assert that the imaginary part is small
    assert np.sum(np.abs(np.imag(rfield))) < 1e-10

    rfield = np.real(rfield)
    
    return jnp.asarray(rfield)

from jax.random import PRNGKey, uniform
import jax

def create_incompressible_turb_field(Ndim, A0, slope, kmin, kmax, seed=None):
    """
    Creates an incompressible turbulent vector field with a given power spectrum.

    Parameters:
        Ndim (int): Dimension of the cubic grid (Ndim x Ndim x Ndim).
        A0 (float): Amplitude scaling factor for the power spectrum.
        slope (float): Slope of the power spectrum (typically -5/3 for Kolmogorov).
        kmin (float): Minimum wavenumber.
        kmax (float): Maximum wavenumber.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (vx, vy, vz), each of shape (Ndim, Ndim, Ndim).
    """
    # Create a random key
    key = PRNGKey(seed if seed is not None else 0)

    # Define the grid in Fourier space
    kx = jnp.fft.fftfreq(Ndim) * Ndim
    ky = jnp.fft.fftfreq(Ndim) * Ndim
    kz = jnp.fft.fftfreq(Ndim) * Ndim
    
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_squared = kx**2 + ky**2 + kz**2

    # Define the power spectrum
    k_magnitude = jnp.sqrt(k_squared)
    power_spectrum = A0 * (k_magnitude**slope)
    power_spectrum = jnp.where((k_magnitude >= kmin) & (k_magnitude <= kmax), power_spectrum, 0.0)

    # Generate random phases
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    phase_vx = uniform(subkey1, shape=(Ndim, Ndim, Ndim), minval=0, maxval=2 * jnp.pi)
    phase_vy = uniform(subkey2, shape=(Ndim, Ndim, Ndim), minval=0, maxval=2 * jnp.pi)
    phase_vz = uniform(subkey3, shape=(Ndim, Ndim, Ndim), minval=0, maxval=2 * jnp.pi)

    # Create Fourier-space velocities with the desired power spectrum and random phases
    vx_k = jnp.sqrt(power_spectrum) * jnp.exp(1j * phase_vx)
    vy_k = jnp.sqrt(power_spectrum) * jnp.exp(1j * phase_vy)
    vz_k = jnp.sqrt(power_spectrum) * jnp.exp(1j * phase_vz)

    # Project to incompressible field (divergence-free)
    k_dot_v = kx * vx_k + ky * vy_k + kz * vz_k
    factor = k_dot_v / (k_squared + 1e-10)  # Avoid division by zero
    vx_k -= factor * kx
    vy_k -= factor * ky
    vz_k -= factor * kz

    # Transform back to real space
    vx = jnp.fft.ifftn(vx_k).real
    vy = jnp.fft.ifftn(vy_k).real
    vz = jnp.fft.ifftn(vz_k).real

    return vx, vy, vz