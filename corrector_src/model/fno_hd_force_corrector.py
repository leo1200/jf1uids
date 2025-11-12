import jax
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, Key
import jax.numpy as jnp

# Simulation framework imports
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams

# Physics utilities
from jf1uids._physics_modules._mhd._vector_maths import curl2D, curl3D
from typing import Optional
import math
from jf1uids.time_stepping._utils import _unpad


class SpectralConv3d(eqx.Module):
    """
    3D spectral convolution in JAX/Equinox.

    - Operates at the input resolution (no spectral upsampling by default).
    - Uses rfftn/irfftn: stores only non-redundant kz >= 0 modes.
    - Splits the (kx, ky) plane into 4 regions (quadrants) and uses separate
      complex weights for each quadrant (weights1..4).
    - shifting_modes >= 0 shifts the window of modes away from DC.
    """

    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    modes3: int
    shifting_modes: int

    # weight arrays (complex64)
    weights1: jnp.ndarray
    weights2: jnp.ndarray
    weights3: jnp.ndarray
    weights4: jnp.ndarray

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int = 0,
        *,
        key: Optional[Key] = None,
    ):
        if key is None:
            key = jax.random.key(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.modes3 = int(modes3)
        self.shifting_modes = int(max(0, shifting_modes))

        # Recommended initialization scale (as discussed)
        scale = 1.0 / math.sqrt(max(1, self.in_channels))

        def _init_complex(key, shape):
            # real + 1j * imag both normal(0,1) scaled
            k_r, k_i = jax.random.split(key)
            r = jax.random.normal(k_r, shape, dtype=jnp.float32)
            i = jax.random.normal(k_i, shape, dtype=jnp.float32)
            c = (scale * (r + 1j * i)).astype(jnp.complex64)
            return c

        wshape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            self.modes3,
        )
        self.weights1 = _init_complex(k1, wshape)
        self.weights2 = _init_complex(k2, wshape)
        self.weights3 = _init_complex(k3, wshape)
        self.weights4 = _init_complex(k4, wshape)

    # complex multiplication shorthand
    def compl_mul3d(self, x_ft_region: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        # x_ft_region: (batch, in_c, kx, ky, kz)
        # w: (in_c, out_c, kx, ky, kz)
        # returns (batch, out_c, kx, ky, kz)
        return jnp.einsum("ixyz,ioxyz->oxyz", x_ft_region, w)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments
        ---------
        x: jnp.ndarray
            shape (batch, in_channels, nx, ny, nz) with real dtype (float32/64).
        Returns
        -------
        out: jnp.ndarray
            shape (batch, out_channels, nx, ny, nz) real-valued output.
        """
        in_c, nx, ny, nz = x.shape
        assert in_c == self.in_channels

        # rfftn keeps kz >= 0 (last axis reduced to nz//2 + 1)
        x_ft = jnp.fft.rfftn(x, axes=(-3, -2, -1)).astype(
            jnp.complex64
        )  # shape: (in_c, nx, ny, nz//2 + 1)

        # compute how many modes we'll actually use (bounded by available spectrum)
        # note: for nx, we treat available kx indices as nx (full signed range), but usable positive half is nx//2
        m1 = int(min(self.modes1, x_ft.shape[-3] // 2))
        m2 = int(min(self.modes2, x_ft.shape[-2] // 2))
        m3 = int(min(self.modes3, x_ft.shape[-1]))

        s = int(self.shifting_modes)
        assert m1 + s <= x_ft.shape[-3] // 2 + 0, "modes1 + shift too large for input"
        assert m2 + s <= x_ft.shape[-2] // 2 + 0, "modes2 + shift too large for input"
        assert m3 + s <= x_ft.shape[-1], "modes3 + shift too large for input"

        # define slices
        kx_pos = slice(s, s + m1)
        ky_pos = slice(s, s + m2)
        kz_pos = slice(s, s + m3)

        if s > 0:
            kx_neg = slice(-m1 - s, -s)
            ky_neg = slice(-m2 - s, -s)
        else:
            kx_neg = slice(-m1, None)
            ky_neg = slice(-m2, None)

        # prepare out FT buffer at the same resolution as if we would call irfftn with the original sizes
        out_ft = jnp.zeros(
            (self.out_channels, nx, ny, nz // 2 + 1), dtype=jnp.complex64
        )

        # Quadrant 1: (+kx, +ky)
        if (m1 > 0) and (m2 > 0) and (m3 > 0):
            out_ft = out_ft.at[..., kx_pos, ky_pos, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_pos, ky_pos, kz_pos], self.weights1[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 2: (-kx, +ky)
            out_ft = out_ft.at[..., kx_neg, ky_pos, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_neg, ky_pos, kz_pos], self.weights2[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 3: (+kx, -ky)
            out_ft = out_ft.at[..., kx_pos, ky_neg, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_pos, ky_neg, kz_pos], self.weights3[..., :m1, :m2, :m3]
                )
            )

            # Quadrant 4: (-kx, -ky)
            out_ft = out_ft.at[..., kx_neg, ky_neg, kz_pos].set(
                self.compl_mul3d(
                    x_ft[..., kx_neg, ky_neg, kz_pos], self.weights4[..., :m1, :m2, :m3]
                )
            )

        # inverse rFFT -> real physical space
        out = jnp.fft.irfftn(out_ft, s=(nx, ny, nz), axes=(-3, -2, -1))

        # output shape: (batch, out_c, nx, ny, nz) real-valued
        return jnp.real(out)


class FNO3dLayer(eqx.Module):
    spectral_conv: SpectralConv3d
    w: jnp.ndarray  # pointwise linear weight (in_c -> out_c)
    b: jnp.ndarray  # pointwise bias

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int = 0,
        *,
        key: Key = None,
    ):
        if key is None:
            key = jax.random.key(0)
        k1, k2 = jax.random.split(key)

        # spectral convolution
        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, modes1, modes2, modes3, shifting_modes, key=k1
        )

        # pointwise linear: weight shape (in_channels, out_channels), bias shape (out_channels,)
        scale = 1.0 / jnp.sqrt(in_channels)
        self.w = scale * jax.random.normal(k2, (in_channels, out_channels))
        self.b = jnp.zeros((out_channels,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (in_channels, nx, ny, nz)
        returns: (out_channels, nx, ny, nz)
        """
        # spectral convolution
        x_spec = self.spectral_conv(x)

        # pointwise linear
        x_lin = jnp.einsum("ixyz,io->oxyz", x, self.w) + self.b[:, None, None, None]

        return x_spec + x_lin


class FNO(eqx.Module):
    layers: eqx.nn.Sequential
    activation: callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_fourier_layers: int,
        fourier_modes: int,
        shifting_modes: int,
        key: Key,
        activation=jax.nn.gelu,
    ):
        self.activation = activation
        keys = jax.random.split(key, n_fourier_layers + 2)
        layers = []

        # Initial Conv3d layer
        layers.append(
            eqx.nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                key=keys[0],
            )
        )

        # Fourier layers with residual connections
        for i in range(n_fourier_layers):
            layers.append(
                FNO3dLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    modes1=fourier_modes,
                    modes2=fourier_modes,
                    modes3=fourier_modes,
                    shifting_modes=shifting_modes,
                    key=keys[i + 1],
                )
            )

        # Final Conv3d layer
        layers.append(
            eqx.nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                key=keys[-1],
            )
        )
        self.layers = eqx.nn.Sequential(layers=layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.layers[0](x)

        for layer in self.layers[1:-1]:
            # Residual connection
            y = layer(out)
            out = self.activation(out + y)
        out = self.layers[-1](out)
        return out


class TurbulenceSGSForceCorrectorFNO(eqx.Module):
    """
    A turbulence subgrid-scale (SGS) force corrector that uses an FNO-based model
    to compute corrections to the primitive state.

    The model learns to predict a correction field ,
    from which a magnetic correction is derived via curl operations.
    """

    fno: FNO
    cambridge_approach: bool = True  # from paper 10.1017/jfm.2022.738
    postprocessing_floor: bool
    output_channels: int

    def __init__(
        self,
        # dimensionality: int,
        # out_channels: int,
        hidden_channels: int,
        n_fourier_layers: int,
        fourier_modes: int,
        shifting_modes: int,
        key: Key,
        postprocessing_floor: bool = False,
        output_channels: int = 3,
    ):
        """Initialize the FNO-based turbulence corrector."""
        if self.cambridge_approach:
            in_channels = 6  # 3 velocities + 3 pressure gradients
        else:
            in_channels = 4  # 3 velocities + presure
            # should i add the density though???
        self.postprocessing_floor = postprocessing_floor
        self.output_channels = output_channels
        self.fno = FNO(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=output_channels,  # for the moment just going to ouput a 3D force field
            n_fourier_layers=n_fourier_layers,
            fourier_modes=fourier_modes,
            shifting_modes=shifting_modes,
            key=key,
        )

    def preprocessing(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
    ) -> Array:
        """Extracts velocity components and pressure gradients."""
        vel_idx = registered_variables.velocity_index
        p_idx = registered_variables.pressure_index
        nc = config.num_ghost_cells
        # Extract velocity field (u, v, w)
        velocity = primitive_state[vel_idx, ...]
        p = primitive_state[p_idx, ...]

        if not self.cambridge_approach:
            # return primitive_state (in case i decide to also return the density)
            return jnp.stack([velocity, p], axis=0)

        # Compute pressure gradients ∇p using central differences
        dx = config.grid_spacing

        grad_p = []
        grad_px = (p[2 * nc :, nc:-nc, nc:-nc] - p[: -2 * nc, nc:-nc, nc:-nc]) / (
            2 * dx
        )
        grad_py = (p[nc:-nc, 2 * nc :, nc:-nc] - p[nc:-nc, : -2 * nc, nc:-nc]) / (
            2 * dx
        )
        grad_pz = (
            (p[nc:-nc, nc:-nc, 2 * nc :] - p[nc:-nc, nc:-nc, : -2 * nc]) / (2 * dx)
            if config.dimensionality == 3
            else jnp.zeros_like(grad_px)
        )
        grad_p = jnp.stack([grad_px, grad_py, grad_pz], axis=0)
        velocity = velocity[:, nc:-nc, nc:-nc, nc:-nc]
        inputs = jnp.concatenate([velocity, grad_p], axis=0)
        return inputs

    def postprocessing(
        self,
        primitive_state: STATE_TYPE,
        registered_variables: RegisteredVariables,
    ):
        if self.postprocessing_floor:
            p_min = 1e-20
            primitive_state = primitive_state.at[
                registered_variables.pressure_index
            ].set(
                jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
            )
            rho_min = 1e-20
            primitive_state = primitive_state.at[
                registered_variables.density_index
            ].set(
                jnp.maximum(
                    primitive_state[registered_variables.density_index], rho_min
                )
            )
        return primitive_state

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ):
        """
        Compute SGS force corrections to the primitive state.

        Args:
            primitive_state: [C, X, Y, Z] array (C = physical variables)
            config: Simulation configuration
            registered_variables: Registered variable indices
            params: Simulation parameters
            time_step: Simulation time step (scalar)
        """

        x = self.preprocessing(
            primitive_state=primitive_state,
            config=config,
            registered_variables=registered_variables,
        )

        forces = self.fno(x)

        x = self.postprocessing(
            primitive_state=primitive_state, registered_variables=registered_variables
        )
        # === Apply correction ===

        nc = config.num_ghost_cells
        if self.output_channels == 3:
            for i, v_index in enumerate(registered_variables.velocity_index):
                primitive_state = primitive_state.at[
                    v_index, nc:-nc, nc:-nc, nc:-nc
                ].add(forces[i] * time_step)
        elif self.output_channels == 5:
            primitive_state = primitive_state.at[:, nc:-nc, nc:-nc, nc:-nc].add(
                forces * time_step
            )

        return primitive_state
