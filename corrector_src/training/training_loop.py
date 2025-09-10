# from autocvd import autocvd

# autocvd(num_gpus=1)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration

import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states

import numpy as np

import jax.numpy as jnp
import jax

from corrector_src.training.train_one_sim import (
    step_based_training_with_losses,
)


from corrector_src.training.training_config import TrainingConfig
from corrector_src.model._cnn_mhd_corrector import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)

from corrector_src.training.loss import mse_loss

import equinox as eqx
import matplotlib.pyplot as plt
import os
import yaml

config_file = yaml.safe_load(open("config.yaml", "r"))
num_cells_hr = config_file["training"]["hr_res"]
downsampling_factor = config_file["training"]["downscaling_factor"]

epochs = config_file["training"]["epochs"]

# Training configuration
training_config = TrainingConfig(
    compute_intermediate_losses=True,
    n_look_behind=config_file["training"]["n_look_behind"],
    loss_function=mse_loss,
    loss_weights=None,
    use_relative_error=False,
)

model = CorrectorCNN(
    in_channels=8,
    hidden_channels=config_file["training"]["hidden_channels"],
    key=jax.random.PRNGKey(42),
)
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

cnn_mhd_corrector_config = CNNMHDconfig(
    cnn_mhd_corrector=True, network_static=neural_net_static
)

cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

snapshot_losses = []
epoch_losses = []
randomized_vars = [1, 1, 1]
for i in range(epochs):
    (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        _,
    ) = blast.randomized_initial_blast_state(num_cells_hr, randomized_vars)

    config = finalize_config(config, initial_state.shape)

    # print("time integrating")
    final_states_hr = time_integration(
        initial_state, config, params, helper_data, registered_variables
    )

    ground_truth = final_states_hr.states
    ground_truth = downaverage_states(ground_truth, downsampling_factor)
    training_config._replace(ground_truth_snapshots=ground_truth)

    initial_state, config, params, helper_data, registered_variables, _ = (
        blast.randomized_initial_blast_state(
            num_cells_hr // downsampling_factor, randomized_vars
        )
    )

    config = finalize_config(config, initial_state.shape)

    config = config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config)
    params = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    losses, new_network_params, final_full_sim_data = step_based_training_with_losses(
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        training_config,
        ground_truth,
    )
    snapshot_losses.append(losses.flatten())
    epoch_losses.append(np.mean(losses))
    cnn_mhd_corrector_params._replace(network_params=new_network_params)
    print(f"epoch {i}", end="\r")

snapshot_losses = np.array(snapshot_losses)
snapshot_losses = snapshot_losses.flatten()

experiment_folder = os.path.join(
    "experiments", f"hc_{config_file['training']['hidden_channels']}"
)
os.mkdir(experiment_folder)
plt.figure()
plt.plot(snapshot_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(experiment_folder, "snapshot_loss_curve.png"))
plt.close()


plt.figure()
plt.plot(epoch_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(experiment_folder, "epoch_loss_curve.png"))
plt.close()

final_model = eqx.combine(new_network_params, neural_net_static)
eqx.tree_serialise_leaves(os.path.join(experiment_folder, "cnn_model.eqx"), model)
