import numpy as np
import jax.numpy as jnp

# def create_turb_field(Ndim, A0, slope, kmin, kmax, seed = None):
#     """Creates a turbulent field with given slope, amplitude
#        and cutoffs in Fourier space for a uniform grid in 3D.

#     Parameters
#     ----------
#     Ndim: int
#         the number of grid points in each dimension
#     A0: float
#         the amplitude of the field
#     slope: float
#         the slope of the power spectrum
#     kmin: float
#         the minimum wavenumber
#     kmax: float
#         the maximum wavenumber
    
#     Returns
#     -------
#     rfield: np.ndarray
#         the real field
#     """
    
#     # construct the k-vectors
#     # wave number bin centers in cycles per unit
#     # of the sample spacing
#     d = 1
#     k = np.fft.fftfreq(Ndim, d=d)*(d*Ndim)
#     # * (d * Ndim) gets the actual wavenumbers, so
#     # k = [0, 1, 2, ..., Ndim/2-1, -Ndim/2, ..., -1] for Ndim even
#     # k = [0, 1, 2, ..., Ndim/2, -Ndim/2+1, ..., -1] for Ndim odd
#     kx, ky, kz = np.meshgrid(k, k, k)
#     k3d = np.sqrt(kx**2 + ky**2 + kz**2)

#     # create the amplitudes of the power spectrum A0 * k**slope, kmin < k < kmax
#     ampli  = np.zeros((Ndim,Ndim,Ndim), dtype=np.float64)
#     idx = np.where((k3d < kmin) | (k3d > kmax))
#     ampli[idx] = 0.0
#     idx = np.where((k3d >= kmin) & (k3d <= kmax))
#     ampli[idx] = A0 * np.power(k3d[idx],slope)

#     # create random phase for the field in Fourier space
#     if seed is not None:
#         np.random.seed(seed)
    
#     phase = np.random.uniform(low=0.0, high=2.*np.pi, size=Ndim**3).reshape(Ndim, Ndim, Ndim)
#     # construct the fourier field with the given amplitude and phase
#     ffield = ampli*np.cos(phase) + ampli*np.sin(phase)*1j

#     # for a real field v, v* = v in real space, so by
#     # the definition of the Fourier transform
#     # \hat{v}_k = \hat{v}_{-k}^*
#     # we enforce this to have a real velocity field
#     for i in range(Ndim):
#         # print("outer loop iteration {} of {}".format(i, Ndim))
#         for j in range(Ndim):
#             for k in range(Ndim//2 + 1):
#                 ffield[i,j,k] = np.conjugate(ffield[-i,-j,-k])

#     # also as ffield_[0,0,0] = ffield*_[-0,-0,-0], ffield_[0,0,0] must be real
#     # likewise as of aliasing
#     # ffield_[Ndim//2, Ndim//2, Ndim//2] = ffield*_[-Ndim//2, -Ndim//2, -Ndim//2] = ffield*_[Ndim//2, Ndim//2, Ndim//2]
#     # by the same reasoning
#     # ffield_[Ndim//2, Ndim//2, 0], ffiedl_[Ndim//2, 0, Ndim//2], ffield_[0, Ndim//2, Ndim//2] must be real
#     # and
#     # ffield_[Ndim//2, 0, 0], ffield_[0, Ndim//2, 0], ffield_[0, 0, Ndim//2] must be real
#     # in these cases we just take the absolute value of the complex number
#     ffield[Ndim//2, Ndim//2, Ndim//2] = np.abs(ffield[Ndim//2, Ndim//2, Ndim//2])
#     ffield[Ndim//2, Ndim//2, 0] = np.abs(ffield[Ndim//2, Ndim//2, 0])
#     ffield[Ndim//2, 0, Ndim//2] = np.abs(ffield[Ndim//2, 0, Ndim//2])
#     ffield[0, Ndim//2, Ndim//2] = np.abs(ffield[0, Ndim//2, Ndim//2])
#     ffield[Ndim//2, 0, 0] = np.abs(ffield[Ndim//2, 0, 0])
#     ffield[0, Ndim//2, 0] = np.abs(ffield[0, Ndim//2, 0])
#     ffield[0, 0, Ndim//2] = np.abs(ffield[0, 0, Ndim//2])
#     ffield[0, 0, 0] = np.abs(ffield[0, 0, 0])

#     # get the real field
#     rfield = np.fft.ifftn(ffield)

#     # assert that the imaginary part is small
#     assert np.sum(np.abs(np.imag(rfield))) < 1e-10

#     rfield = np.real(rfield)
    
#     return jnp.asarray(rfield)

# def create_turb_field(Ndim, A0, slope, kmin, kmax, seed=None):
#     """Creates a turbulent field with given slope, amplitude
#        and cutoffs in Fourier space for a uniform grid in 3D.
#     """
#     d = 1.0
#     k = jnp.fft.fftfreq(Ndim, d=d) * (d * Ndim)
#     kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
#     k3d = jnp.sqrt(kx**2 + ky**2 + kz**2)

#     # amplitudes
#     ampli = jnp.where(
#         (k3d >= kmin) & (k3d <= kmax),
#         A0 * jnp.power(k3d, slope),
#         0.0,
#     )

#     # random phase
#     key = jax.random.PRNGKey(0 if seed is None else seed)
#     phase = jax.random.uniform(key, shape=(Ndim, Ndim, Ndim), minval=0.0, maxval=2.0 * jnp.pi)

#     # Fourier field with Hermitian symmetry
#     ffield = ampli * (jnp.cos(phase) + 1j * jnp.sin(phase))

#     # enforce Hermitian symmetry automatically
#     # by averaging with the conjugate of the reversed array
#     ffield = 0.5 * (ffield + jnp.conjugate(jnp.flip(jnp.flip(jnp.flip(ffield, 0), 1), 2)))

#     # fix special Nyquist frequencies (must be real)
#     nyq = Ndim // 2
#     special_indices = [
#         (nyq, nyq, nyq),
#         (nyq, nyq, 0),
#         (nyq, 0, nyq),
#         (0, nyq, nyq),
#         (nyq, 0, 0),
#         (0, nyq, 0),
#         (0, 0, nyq),
#         (0, 0, 0),
#     ]
#     for idx in special_indices:
#         ffield = ffield.at[idx].set(jnp.abs(ffield[idx]))

#     # inverse FFT to get real field
#     rfield = jnp.fft.ifftn(ffield)
#     rfield = jnp.real(rfield)

#     return rfield


def create_turb_field(Ndim, A0, slope, kmin, kmax, key, sharding = None):
    """Creates a turbulent field with a given power spectrum in JAX.

    This function generates a 3D real-valued field with a power spectrum
    P(k) ~ k^slope within a specified range of wavenumbers.

    Parameters
    ----------
    Ndim: int
        The number of grid points in each dimension.
    A0: float
        The amplitude constant of the power spectrum.
    slope: float
        The slope of the power spectrum, P(k) ~ A0^2 * k^(2*slope).
        Note: This is the slope for the field amplitude, not power.
    kmin: float
        The minimum wavenumber for the power spectrum.
    kmax: float
        The maximum wavenumber for the power spectrum.
    key: jax.random.PRNGKey
        The JAX random key for reproducibility.

    Returns
    -------
    rfield: jax.Array
        The real-valued turbulent field in configuration space.
    """
    # 1. Construct the k-vectors in 3D
    # JAX's fftfreq is equivalent to numpy's. We multiply by Ndim to get
    # integer wavenumbers [0, 1, ..., N/2, -N/2+1, ..., -1]
    k = jnp.fft.fftfreq(Ndim, d=1.0) * Ndim
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij')

    if sharding is not None:
        kx = jax.device_put(kx, sharding)
        ky = jax.device_put(ky, sharding)
        kz = jax.device_put(kz, sharding)

    k3d = jnp.sqrt(kx**2 + ky**2 + kz**2)

    # 2. Define the power spectrum amplitude in Fourier space
    # Create a mask for the wavenumbers within the desired range
    mask = (k3d >= kmin) & (k3d <= kmax)
    
    # Calculate the amplitude A0 * k^slope
    # We guard against k=0 for negative slopes to avoid division by zero
    k3d_safe = jnp.where(k3d == 0, 1.0, k3d)
    power_law = A0 * jnp.power(k3d_safe, slope)
    
    # Apply the mask to get the final amplitude
    ampli = jnp.where(mask, power_law, 0.0)

    # 3. Create a random field in Fourier space with the given amplitude
    # Generate random phases
    phase = jax.random.uniform(key, shape=k3d.shape, minval=0.0, maxval=2. * jnp.pi)
    if sharding is not None:
        phase = jax.device_put(phase, sharding)
    
    # Construct the complex Fourier field
    ffield_random = ampli * jnp.exp(1j * phase)

    # 4. Enforce Hermitian symmetry to ensure the field is real in configuration space.
    # The condition is F(k) = F*(-k), where F* is the complex conjugate.
    # We do this by averaging the random field with its flipped, conjugated version.
    
    # Generate indices for all points in the grid
    i, j, k_idx = jnp.indices((Ndim, Ndim, Ndim))
    
    # Calculate the indices corresponding to -k. In a periodic grid of size N,
    # the index for -i is (N-i) % N, which is equivalent to -i % N.
    ni, nj, nk = (-i % Ndim), (-j % Ndim), (-k_idx % Ndim)
    
    # Create the flipped and conjugated field
    ffield_flipped_conj = jnp.conj(ffield_random[ni, nj, nk])
    
    # The new field is the average of the two, which now satisfies symmetry.
    # This also correctly handles the DC (0,0,0) and Nyquist frequencies,
    # forcing them to be real, as F(k)=F*(-k) implies F(k) must be real if k=-k.
    ffield = 0.5 * (ffield_random + ffield_flipped_conj)

    # 5. Transform back to real space
    rfield_complex = jnp.fft.ifftn(ffield)

    # The imaginary part should be negligible due to the enforced symmetry.
    # We return the real part.
    rfield = jnp.real(rfield_complex)
    
    return rfield

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