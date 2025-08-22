from autocvd import autocvd
autocvd(num_gpus=1)

import corrector_src.data.blast_creation as blast
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

num_cells = 64
fig, axes = plt.subplots(2, 5, figsize = (25, 10))

for ax in axes.flat:
    initial_state, config, params, helper_data, registered_variables, randomized_variables = blast.randomized_initial_blast_state(num_cells)
    ax.imshow(initial_state[4, :, : , 32], norm=plt.Normalize(vmin=0, vmax=1))
    ax.set_title(f'r_inj {randomized_variables[0]:.3f}, p_inj {randomized_variables[1]:.3f}, B_0 {randomized_variables[2]:.3f}')
    
fig.savefig("figures/randomized_states.png")