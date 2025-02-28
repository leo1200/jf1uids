# Introduction and Installation

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

self

```

`jf1uids` differentiable (magneto)hydrodynamics (MHD) for astrophysics written in `JAX`.


::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Fast and Differentiable

Written in `JAX`, `jf1uids` is fully differentiable - a simulation can be differentiated with respect to any input parameter - and just-in-time compiled for fast execution on CPU, GPU, or TPU.

+++
[Learn more »](notebooks/stellar_wind/gradients_through_stellar_wind.ipynb)
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Well-considered Numerical Methods

Divergence-free MHD based on [Pang and Wu (2024)](https://arxiv.org/abs/2410.05173) and conservative radial simulations based on [Crittenden and Balachandar (2018)](https://doi.org/10.1007/s00193-017-0784-y).

+++
[Learn more »](notebooks/magnetohydrodynamics/orszag_tang_vortex.ipynb)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Extensible

Physical modules, like one for stellar winds, can be easily added to the simulation. These modules can include parameters (or neural network terms) which can be optimized directly in the solver. Further primitive variables,
like those for a [cosmic ray fluid](notebooks/cosmic_rays/simple_example_cr.ipynb), are already implemented or can be added.

+++
[Learn more »](notebooks/stellar_wind/wind_parameter_optimization.ipynb)
:::

::::

:::{seealso}
The corresponding paper for the previous 1d-only version of this project is available on [arXiv](https://arxiv.org/abs/2410.23093).
:::

## Installation

`jf1uids` can be installed via `pip`

```bash
pip install jf1uids
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

:::{tip} Get started with this [simple example](notebooks/hydrodynamics/simple_example.ipynb).
:::

## Showcase

| ![Orszag-Tang Vortex](notebooks/figures/orszag_tang_animation.gif) | ![3D Collapse](notebooks/figures/3d_collapse.gif) |
|:------------------------------------------------------------------:|:-------------------------------------------------:|
| Orszag-Tang Vortex                                                 | 3D Collapse                                       |

| ![Gradients Through Stellar Wind](notebooks/figures/gradients_through_stellar_wind.svg) |
|:---------------------------------------------------------------------------------------:|
| Gradients Through Stellar Wind                                                          |

| ![Wind Parameter Optimization](notebooks/figures/wind_parameter_optimization.png) |
|:---------------------------------------------------------------------------------:|
| Wind Parameter Optimization                                                       |

## Notebooks for Getting Started

```{toctree}
:caption: hydrodynamics
:maxdepth: 1

notebooks/hydrodynamics/simple_example.ipynb
notebooks/hydrodynamics/conservational_properties.ipynb
notebooks/hydrodynamics/kelvin_helmholtz.ipynb
```

```{toctree}
:caption: magnetohydrodynamics
:maxdepth: 1

notebooks/magnetohydrodynamics/orszag_tang_vortex.ipynb
```

```{toctree}
:caption: self-gravity
:maxdepth: 1

notebooks/self_gravity/evrards_collapse.ipynb
```

```{toctree}
:caption: stellar wind
:maxdepth: 1

notebooks/stellar_wind/gradients_through_stellar_wind.ipynb
notebooks/stellar_wind/wind_parameter_optimization.ipynb
notebooks/stellar_wind/stellar_wind3D.ipynb
```

## Roadmap

:::{seealso}
A baseline hydrodynamical code for dynamic adaptive mesh refinement (AMR) in `JAX` in one dimension, `jamr`, is available on [GitHub](https://github.com/leo1200/jamr).
:::

- [x] Implement a conservative 1d radial fluid solver with simple to implement yet powerful numerical schemes.
- [x] Implement a simple stellar wind model, first analyses on the gradients of the final fluid state with respect to the wind parameters.
- [ ] Implementation of higher-order reconstruction methods like WENO-Z+, etc. 
- [x] Implementation of different Riemann solvers (currently HLL, HLLC, HLLC-LM)
- [x] (advection-only) two-fluid cosmic-ray model
- [ ] cosmic ray diffusion
- [ ] Implementation of a shock finder
- [x] Generalize to 2D and 3D
- [x] implement self-gravity
- [ ] improve self-gravity for energy conservation


```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

source/jf1uids.time_stepping.time_integration

```
