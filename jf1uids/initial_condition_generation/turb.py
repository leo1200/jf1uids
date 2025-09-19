import numpy as np
import jax.numpy as jnp

import jax

def _cosine_taper(k, k0, k1):
    """Returns taper factor in [0,1]: 1 for k<=k0, smooth down to 0 at k>=k1."""
    # If k1 == k0, use a hard cutoff
    small = 1e-12
    if k1 <= k0 + small:
        return jnp.where(k <= k0, 1.0, 0.0)
    t = (k - k0) / (k1 - k0)
    t = jnp.clip(t, 0.0, 1.0)
    return jnp.where(k <= k0, 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * t)))

# k_max <= int(0.7 * Ndim/2)

def create_turb_field(
    Ndim,
    A0,
    slope,
    kmin,
    kmax,
    key,
    sharding=None,
    kroll_frac=0.85,
    zero_mean=True,
):
    """Generate a real Gaussian random field with target amplitude scaling.

    Improvements vs original:
    - samples complex Gaussian coefficients (not constant magnitude),
    - cosine roll-off near kmax to avoid Nyquist injection,
    - zeros DC by default,
    - enforces Hermitian symmetry and real-valued self-conj modes.
    """
    # build integer wavenumbers [-N/2..N/2-1] pattern via fftfreq*N
    k1d = jnp.fft.fftfreq(Ndim, d=1.0) * Ndim
    kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing="ij")
    if sharding is not None:
        kx = jax.device_put(kx, sharding)
        ky = jax.device_put(ky, sharding)
        kz = jax.device_put(kz, sharding)

    k3d = jnp.sqrt(kx**2 + ky**2 + kz**2)

    # optional roll-off: start rolling at kroll_frac * kmax and finish at kmax
    k_roll_start = kroll_frac * kmax
    taper = _cosine_taper(k3d, k_roll_start, kmax)

    # mask inside k-band where we want power (we will further multiply by taper)
    band_mask = (k3d >= kmin) & (k3d <= kmax)

    # amplitude: note we set amplitude to 0 at k=0 (unless user wants mean)
    # Interpret `slope` as exponent for amplitude (A ~ k^slope). If you wanted power ~ k^p,
    # then amplitude exponent = p/2.
    # set k_safe but leave k=0 special so we can zero it out
    k_safe = jnp.where(k3d == 0.0, 1.0, k3d)
    amplitude = A0 * (k_safe ** slope)
    if zero_mean:
        amplitude = amplitude.at[0, 0, 0].set(0.0)

    # apply band + taper
    amplitude = amplitude * band_mask * taper

    # For a Gaussian field: set complex Fourier coef with variance ~ amplitude^2
    # We want E[|F_k|^2] proportional to amplitude^2. For complex Gaussian
    # with independent real & imag of variance sigma^2, E[|F|^2]=2*sigma^2.
    # So choose sigma = amplitude / sqrt(2).
    sigma = amplitude / jnp.sqrt(2.0)

    # sample real and imaginary parts ~ Normal(0, sigma^2)
    subkeys = jax.random.split(key, 2)
    re = jax.random.normal(subkeys[0], shape=k3d.shape) * sigma
    im = jax.random.normal(subkeys[1], shape=k3d.shape) * sigma
    F = re + 1j * im

    # Enforce Hermitian symmetry: F(-k) = conj(F(k))
    # We'll do symmetric averaging and then explicitly set self-conj indices to be real.
    i, j, k_idx = jnp.indices((Ndim, Ndim, Ndim))
    ni, nj, nk = (-i) % Ndim, (-j) % Ndim, (-k_idx) % Ndim
    F_sym = 0.5 * (F + jnp.conj(F[ni, nj, nk]))

    # find self-conjugate indices: where (i,j,k) == (-i,-j,-k) mod N
    self_conj = (i == ni) & (j == nj) & (k_idx == nk)
    # force imaginary part zero at these locations (they must be real)
    F_sym = jnp.where(self_conj, jnp.real(F_sym).astype(F_sym.dtype), F_sym)

    # optional: explicitly ensure DC is real zero (if zero_mean True)
    if zero_mean:
        F_sym = F_sym.at[0, 0, 0].set(0.0 + 0.0j)

    # inverse transform back to real space
    rfield_complex = jnp.fft.ifftn(F_sym)
    rfield = jnp.real(rfield_complex)

    return rfield