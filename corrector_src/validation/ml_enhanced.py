# from autocvd import autocvd

# autocvd(num_gpus=1)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd

"""
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration

import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states
from corrector_src.training.loss import mse_loss

import equinox as eqx
import matplotlib.pyplot as plt
import os
import random
"""

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration


import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states
from corrector_src.model._cnn_mhd_corrector import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)

import jax
import equinox as eqx
import random
import yaml
from itertools import product
import numpy as np

config_file = yaml.safe_load(open("config.yaml", "r"))

n_tries = 1
num_cells_hr = 64
downscaling = 2

randomizers_grid = {
    "rand_1": np.linspace(
        config_file["blast_creation"]["randomizer_1"][0],
        config_file["blast_creation"]["randomizer_1"][0],
        n_tries,
    ),
    " rand_2": np.linspace(
        config_file["blast_creation"]["randomizer_2"][0],
        config_file["blast_creation"]["randomizer_2"][1],
        n_tries,
    ),
    "rand_3": np.linspace(
        config_file["blast_creation"]["randomizer_3"][0],
        config_file["blast_creation"]["randomizer_3"][1],
        n_tries,
    ),
}

keys = list(randomizers_grid.keys())
grid = list(product(*randomizers_grid.values()))


model = CorrectorCNN(
    in_channels=8,
    hidden_channels=32,
    key=jax.random.PRNGKey(42),
)
model = eqx.tree_deserialise_leaves("model/cnn_model.eqx", model)

neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

cnn_mhd_corrector_config = CNNMHDconfig(
    cnn_mhd_corrector=True, network_static=neural_net_static
)

cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)


def sim_downscale(randomizers):
    (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        randomized_variables,
    ) = blast.randomized_initial_blast_state(num_cells_hr, randomizers)

    config = finalize_config(config, initial_state.shape)

    final_states_hr = time_integration(
        initial_state, config, params, helper_data, registered_variables
    )

    ground_truth = final_states_hr.states
    ground_truth_lr = downaverage_states(ground_truth, downscaling)
    return ground_truth_lr


def sim(randomizers):
    (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        randomized_variables,
    ) = blast.randomized_initial_blast_state(num_cells_hr // downscaling, randomizers)

    config = finalize_config(config, initial_state.shape)

    final_states = time_integration(
        initial_state,
        config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config),
        params._replace(
            cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                network_params=neural_net_params
            )
        ),
        helper_data,
        registered_variables,
    )

    final_states = final_states.states
    return final_states


def loss(downscaled_states, lr_states):
    return np.mean((downscaled_states - lr_states) ** 2)


results = []
for randomizers in grid:
    ds_states = sim_downscale(randomizers)
    lr_states = sim(randomizers)
    results.append(
        {
            "rand_0": randomizers[0],
            "rand_1": randomizers[1],
            "rand_2": randomizers[2],
            "loss": loss(ds_states, lr_states),
        }
    )
    print(results[-1], end="\r")

df = pd.DataFrame(results)
df.to_csv(os.path.join("validation", "ml_enhanced_loss.csv"), index=False)
