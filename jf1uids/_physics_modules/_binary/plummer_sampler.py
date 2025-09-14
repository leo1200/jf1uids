"""
Plummer sampler for N-body initial conditions (3D).

Generates positions from a Plummer sphere and samples velocities from the
local Plummer velocity distribution using rejection sampling. It then
rescales velocities to reach a target virial ratio Q (default 0.5) and
optionally enforces a confinement box [-a,a]^3 by a similarity rescale.

Returns:
    txv : jnp.array shape (n,7) -- rows [t, x, y, z, vx, vy, vz]
    masses : jnp.array shape (n,) -- masses for each particle

Uses numpy for the sampler and converts results to jax.numpy (jnp)
for the requested output format.

Notes:
- G defaults to 1.0. If you use a different G in your integrator, either
  change G here or scale positions/velocities consistently.
- The Plummer scale `b` defaults to a/2 so that most mass lies well
  within the box radius `a`. Change `b` if you want a more/less
  concentrated distribution.

Example:
    txv, m = plummer_sampler(n=100, M1=0.8, M2=1.2, a=1.0, seed=42)
"""

from typing import Tuple, Optional
import numpy as np
import jax.numpy as jnp


def plummer_sampler(
    n: int,
    M1: float,
    M2: float,
    a: float,
    b: Optional[float] = None,
    G: float = 1.0,
    Q: float = 0.5,
    seed: Optional[int] = None,
    t0: float = 0.0,
    masses: Optional[np.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Plummer initial conditions and return txv & masses.

    Parameters
    ----------
    n : int
        Number of particles.
    M1, M2 : float
        Min/max mass (uniform sampling in [M1, M2]) unless `masses` is given.
    a : float
        Half-box size: coordinates will be in [-a,a]^3 after an optional
        similarity rescale.
    b : Optional[float]
        Plummer scale radius. If None, defaults to `a/2`.
    G : float
        Gravitational constant.
    Q : float
        Target virial ratio T/|U| after velocity rescaling (default 0.5).
    seed : Optional[int]
        RNG seed for reproducibility.
    t0 : float
        Time assigned to every row in txv (default 0.0).
    masses : Optional[np.ndarray]
        If provided, used as the masses array (length n). Otherwise samples
        uniformly from [M1, M2].

    Returns
    -------
    txv : jnp.ndarray shape (n,7)
        Rows: [t, x, y, z, vx, vy, vz]
    masses : jnp.ndarray shape (n,)

    """

    rng = np.random.default_rng(seed)

    if masses is None:
        masses_np = rng.uniform(low=M1, high=M2, size=n).astype(float)
    else:
        masses_np = np.asarray(masses, dtype=float).copy()
        assert masses_np.shape == (n,)

    M_tot = masses_np.sum()

    # --- Plummer scale
    if b is None:
        b = a / 2.0

    # --- sample radii (Plummer) and directions
    # r = b * (u^{-2/3} - 1)^{-1/2}
    u = rng.random(n)
    r = b * (u ** (-2.0 / 3.0) - 1.0) ** (-0.5)

    # random isotropic directions
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    # --- local escape speeds
    psi = G * M_tot / np.sqrt(r * r + b * b)  # relative potential
    v_esc = np.sqrt(2.0 * psi)

    # --- prepare rejection sampling for the reduced speed f = v / v_esc
    # pdf(f) propto f^2 * (1 - f^2)^{7/2} for f in [0,1]
    def sample_f_rejection(num_samples: int) -> np.ndarray:
        # precompute p_max by sampling a fine grid (cheap)
        fgrid = np.linspace(0.0, 1.0, 10001)[1:-1]
        pgrid = fgrid ** 2 * (1.0 - fgrid ** 2) ** 3.5
        p_max = pgrid.max()

        samples = np.empty(num_samples, dtype=float)
        i = 0
        while i < num_samples:
            # propose
            f_prop = rng.random(num_samples - i)
            u_prop = rng.random(num_samples - i)
            p_prop = f_prop ** 2 * (1.0 - f_prop ** 2) ** 3.5
            accept = u_prop * p_max <= p_prop
            n_accept = accept.sum()
            if n_accept > 0:
                samples[i : i + n_accept] = f_prop[accept]
                i += n_accept
        return samples

    f_samples = sample_f_rejection(n)

    # random velocity directions (isotropic)
    cos_t_v = rng.uniform(-1.0, 1.0, size=n)
    sin_t_v = np.sqrt(1.0 - cos_t_v ** 2)
    phi_v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    vx_dir = sin_t_v * np.cos(phi_v)
    vy_dir = sin_t_v * np.sin(phi_v)
    vz_dir = cos_t_v

    vx = f_samples * v_esc * vx_dir
    vy = f_samples * v_esc * vy_dir
    vz = f_samples * v_esc * vz_dir

    # --- compute pairwise potential energy U and kinetic T
    coords = np.vstack([x, y, z]).T  # (n,3)
    # pairwise distances
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dz = coords[:, None, 2] - coords[None, :, 2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    # avoid self division; set diagonal to inf
    np.fill_diagonal(dist, np.inf)

    # pairwise mass products
    mprod = masses_np[:, None] * masses_np[None, :]

    U = -0.5 * G * np.sum(mprod / dist)  # factor 1/2 because we sum i!=j both

    speeds2 = vx * vx + vy * vy + vz * vz
    T = 0.5 * np.sum(masses_np * speeds2)

    # --- rescale velocities to achieve virial ratio Q = T / |U|
    if T <= 0.0:
        raise RuntimeError("Kinetic energy nonpositive during sampling.")
    desired_T = Q * abs(U)
    scale_v = np.sqrt(desired_T / T)
    vx *= scale_v
    vy *= scale_v
    vz *= scale_v

    # subtract center-of-mass velocity so the CM is at rest
    v_cm = np.sum(masses_np[:, None] * np.vstack([vx, vy, vz]).T, axis=0) / M_tot
    vx -= v_cm[0]
    vy -= v_cm[1]
    vz -= v_cm[2]

    # --- optionally enforce box confinement by similarity rescale
    max_coord = np.max(np.abs(np.vstack([x, y, z])))
    if max_coord > a:
        lam = a / max_coord  # scale < 1
        x *= lam
        y *= lam
        z *= lam
        # to preserve dynamics, velocities scale as 1/sqrt(lam)
        vel_scale = 1.0 / np.sqrt(lam)
        vx *= vel_scale
        vy *= vel_scale
        vz *= vel_scale

    # final txv array
    txv_np = np.column_stack(
        [np.full(n, t0), x, y, z, vx, vy, vz]
    )  # shape (n,7)

    # convert to jax arrays for the requested output format
    txv = jnp.array(txv_np)
    masses_jnp = jnp.array(masses_np)

    return txv, masses_jnp


"""
Simple virialized sphere sampler for N-body initial conditions (3D).

Generates `n` bodies with masses in [M1, M2] placed inside a sphere of
radius R<=a and assigns velocities so the system has a target virial
ratio Q = T/|U| (default Q=0.5). Positions are isotropic (uniform in
sphere) and velocity directions are isotropic; speeds are initially
drawn from a Maxwell-like distribution and then globally rescaled to
achieve the virial ratio.

Returns:
    txv : jnp.array shape (n,7) -- rows [t, x, y, z, vx, vy, vz]
    masses : jnp.array shape (n,) -- masses for each particle

Notes:
- Uses numpy for sampling and converts outputs to jax.numpy (jnp).
- G defaults to 1.0. If you want a different gravitational constant,
  set G accordingly or rescale positions/velocities consistently.

Example:
    txv, m = virialized_sphere_sampler(100, 0.8, 1.2, a=1.0, R=0.9, seed=1)

"""

from typing import Tuple, Optional
import numpy as np
import jax.numpy as jnp


def virialized_sphere_sampler(
    n: int,
    M1: float,
    M2: float,
    a: float,
    R: Optional[float] = None,
    G: float = 1.0,
    Q: float = 0.5,
    seed: Optional[int] = None,
    t0: float = 0.0,
    masses: Optional[np.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a simple virialized sphere of `n` particles.

    Parameters
    ----------
    n : int
        Number of particles.
    M1, M2 : float
        Min/max mass for uniform sampling in [M1, M2], unless `masses` is
        supplied (length n).
    a : float
        Half-box size. Positions will be confined to [-a,a]^3 (via a
        similarity rescale if needed).
    R : Optional[float]
        Sphere radius to place particles in. If None, defaults to 0.9*a.
    G : float
        Gravitational constant (default 1.0).
    Q : float
        Target virial ratio T/|U| (default 0.5).
    seed : Optional[int]
        RNG seed for reproducibility.
    t0 : float
        Time value for each txv row (default 0.0).
    masses : Optional[np.ndarray]
        If provided, used as masses (must have length n).

    Returns
    -------
    txv : jnp.ndarray shape (n,7)
    masses_jnp : jnp.ndarray shape (n,)

    """

    rng = np.random.default_rng(seed)

    # --- masses
    if masses is None:
        masses_np = rng.uniform(low=M1, high=M2, size=n).astype(float)
    else:
        masses_np = np.asarray(masses, dtype=float).copy()
        if masses_np.shape != (n,):
            raise ValueError("masses must have shape (n,)")

    M_tot = masses_np.sum()

    # --- choose radius
    if R is None:
        R = 0.9 * a

    # --- sample positions uniformly in sphere of radius R
    u = rng.random(n)
    r = R * u ** (1.0 / 3.0)  # invert CDF for uniform sphere
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    # --- initial velocity magnitudes: Maxwell-like draw using sigma~sqrt(G*M_tot/R)
    v_scale = np.sqrt(np.abs(G * M_tot / max(R, 1e-12)))
    # draw three independent normals and take magnitude
    vx0 = rng.normal(scale=v_scale, size=n)
    vy0 = rng.normal(scale=v_scale, size=n)
    vz0 = rng.normal(scale=v_scale, size=n)

    # now we will rescale speeds to reach target virial ratio
    # compute potential energy U
    coords = np.vstack([x, y, z]).T  # (n,3)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dz = coords[:, None, 2] - coords[None, :, 2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    np.fill_diagonal(dist, np.inf)

    mprod = masses_np[:, None] * masses_np[None, :]
    U = -0.5 * G * np.sum(mprod / dist)

    speeds2 = vx0 * vx0 + vy0 * vy0 + vz0 * vz0
    T = 0.5 * np.sum(masses_np * speeds2)

    if T <= 0.0:
        raise RuntimeError("Initial kinetic energy non-positive; try different RNG/params")

    desired_T = Q * np.abs(U)
    scale_v = np.sqrt(desired_T / T)

    vx = vx0 * scale_v
    vy = vy0 * scale_v
    vz = vz0 * scale_v

    # subtract center-of-mass velocity so CM is at rest
    v_cm = np.sum(masses_np[:, None] * np.vstack([vx, vy, vz]).T, axis=0) / M_tot
    vx -= v_cm[0]
    vy -= v_cm[1]
    vz -= v_cm[2]

    # ensure confinement in box [-a,a]^3; if coordinates exceed a, similarity rescale
    max_coord = np.max(np.abs(np.vstack([x, y, z])))
    if max_coord > a:
        lam = a / max_coord
        x *= lam
        y *= lam
        z *= lam
        # velocities must scale as 1/sqrt(lam) to preserve dynamical similarity
        vel_scale = 1.0 / np.sqrt(lam)
        vx *= vel_scale
        vy *= vel_scale
        vz *= vel_scale

    txv_np = np.column_stack([np.full(n, t0), x, y, z, vx, vy, vz])

    txv = jnp.array(txv_np)
    masses_jnp = jnp.array(masses_np)

    return txv, masses_jnp
