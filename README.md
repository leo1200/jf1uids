# jf1uids - Differentiable Conservative Radially Symmetric Fluid Simulations and Stellar Winds

Welcome to the repository of jf1uids, a 1d radial fluid solver written in JAX.

jf1uid is written in a way such that mass and energy are conserved based on the approach of [Crittenden and Balachandar (2018)](https://doi.org/10.1007/s00193-017-0784-y) with a MUSCL-Hancock fluid solver. Note that for any general pupose (non-radial) problem consider using [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS).

The solver's code can be found in the jf1uis folder. You can find a simple example for getting
started under [simple_example.ipynb](simple_example.ipynb).

## Installation
jf1uids can be installed via pip:

```bash
pip install jf1uids
```

For running the notebooks you will also need to install matplotlib (and jupyter).

## Reproducing the results from the paper

All the results from the paper can easily be reproduced via the notebooks provided

- figure 1 &rarr; [conservational_properties.ipynb](notebooks/conservational_properties.ipynb): For a radial shock problem, conservation of mass and energy in jf1uids are showcased.
- figure 2 &rarr; [gradients_through_stellar_wind.ipynb](notebooks/gradients_through_stellar_wind.ipynb): The gradients of the final fluid state with respect to the wind's velocity are analyzed.
- figure 3 &rarr; [wind_parameter_optimization.ipynb](notebooks/wind_parameter_optimization.ipynb): Finding wind parameters from the final fluid state via
gradient-descent is shown.

## Roadmap

- [x] Implement a conservative 1d radial fluid solver with simple to implement yet powerful numerical schemes.
- [x] Implement a simple stellar wind model, first analyses on the gradients of the final fluid state with respect to the wind parameters.
- [ ] Implementation of higher-order reconstruction methods like WENO-Z+, etc. 
- [ ] Implementation of different Riemann solvers
- [ ] Implementation of a shock finder
- [ ] Implementation of a simple cosmic ray model based on the energy dissipation in shocks
- [ ] Two-fluid cosmic-ray model
- [ ] Full-spectrum cosmic ray modeling