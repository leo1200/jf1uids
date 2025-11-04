import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["BoxSize", "axis"])
def pk_jax_1d(delta: jnp.ndarray, BoxSize: float = 1.0, axis: int = 2):
    """
    Compute 1D power spectrum along a specified axis using JAX.

    Parameters:
    -----------
    delta : jnp.ndarray
        3D density field
    BoxSize : float
        Size of the simulation box
    axis : int
        Axis along which to compute parallel modes (0=x, 1=y, 2=z)

    Returns:
    --------
    k1D : jnp.ndarray
        Wavenumbers
    Pk1D : jnp.ndarray
        Power spectrum values
    Nmodes1D : jnp.ndarray
        Number of modes in each bin
    """
    dims = delta.shape[0]
    middle = dims // 2
    kF = 2.0 * jnp.pi / BoxSize
    kN = middle * kF
    kmax_par = middle

    # --- FFT of the field ---
    delta_k = jnp.fft.fftn(delta)

    # Build integer frequency grids matching Cython code
    # The Cython code uses: kx = (kxx-dims if (kxx>middle) else kxx)
    # This creates frequencies in range [-middle, middle] for even dims
    kx_indices = jnp.arange(dims)
    ky_indices = jnp.arange(dims)
    kz_indices = jnp.arange(dims)

    kx = jnp.where(kx_indices > middle, kx_indices - dims, kx_indices)
    ky = jnp.where(ky_indices > middle, ky_indices - dims, ky_indices)
    kz = jnp.where(kz_indices > middle, kz_indices - dims, kz_indices)

    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # ---- Build independent modes mask (matching Cython logic) ----
    mask = jnp.ones_like(KX, dtype=bool)

    # The Cython code only loops over kzz in range(middle+1), effectively kz >= 0
    # So we only keep kz >= 0 modes
    mask = jnp.where(KZ < 0, False, mask)

    # Handle special planes for Hermitian symmetry
    # kz=0 plane: exclude kx < 0
    kz_zero = KZ == 0
    mask = jnp.where(kz_zero & (KX < 0), False, mask)

    # kz=0 or kz=middle plane with kx=0 or kx=middle: exclude ky < 0
    kz_middle = (KZ == middle) & (dims % 2 == 0)
    special_plane = kz_zero | kz_middle
    kx_special = (KX == 0) | ((KX == middle) & (dims % 2 == 0))
    mask = jnp.where(special_plane & kx_special & (KY < 0), False, mask)

    # Keep only modes with |k| <= middle
    k_mag = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    mask = jnp.where(k_mag > middle, False, mask)

    # Compute k_par based on chosen axis
    if axis == 0:
        k_par = jnp.abs(KX)
    elif axis == 1:
        k_par = jnp.abs(KY)
    else:  # axis == 2
        k_par = jnp.abs(KZ)

    # Compute |delta_k|^2
    delta2 = jnp.abs(delta_k) ** 2

    # Apply mask
    delta2_masked = jnp.where(mask, delta2, 0.0)
    k_par_masked = jnp.where(mask, k_par, 0)
    mask_float = mask.astype(jnp.float64)

    # Bin by k_par using vmap approach
    def bin_sum(i):
        in_bin = (k_par_masked == i) & mask
        return jnp.sum(jnp.where(in_bin, delta2_masked, 0.0))

    def bin_k_sum(i):
        in_bin = (k_par_masked == i) & mask
        return jnp.sum(jnp.where(in_bin, k_par * mask_float, 0.0))

    def bin_count(i):
        in_bin = (k_par_masked == i) & mask
        return jnp.sum(in_bin.astype(jnp.float64))

    # Compute for all bins
    indices = jnp.arange(kmax_par + 1)
    Pk1D = jax.vmap(bin_sum)(indices)
    k1D = jax.vmap(bin_k_sum)(indices)
    Nmodes1D = jax.vmap(bin_count)(indices)

    # Remove DC mode (k_par = 0)
    k1D = k1D[1:]
    Pk1D = Pk1D[1:]
    Nmodes1D = Nmodes1D[1:]

    # Compute average k in each bin and apply units
    k1D = jnp.where(Nmodes1D > 0, k1D / Nmodes1D, 0.0) * kF

    # Compute perpendicular sampling area (matching Cython logic)
    # kmaxper = sqrt(kN^2 - k1D^2), where kN is the Nyquist frequency
    kmaxper = jnp.sqrt(jnp.maximum(kN**2 - k1D**2, 0.0))

    # Normalize power spectrum exactly as in Cython code:
    # Pk1D[i] = Pk1D[i]*(BoxSize/dims**2)**3 * (pi*kmaxper^2/Nmodes) / (2*pi)^2
    Pk1D = jnp.where(
        Nmodes1D > 0,
        Pk1D
        * (BoxSize / dims**2) ** 3
        * (jnp.pi * kmaxper**2 / Nmodes1D)
        / (2.0 * jnp.pi) ** 2,
        0.0,
    )
    return k1D, Pk1D, Nmodes1D
