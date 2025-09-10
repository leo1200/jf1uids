from autocvd import autocvd

autocvd(num_gpus=1)


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

num_cells_hr = 64
downsampling_factor = 2


# Load ground truth data (TO BE CHANGED)
def load_ground_truth(filepath="ground_truth.npy"):
    """
    Load the ground truth array from a saved numpy file.

    Args:
        filepath (str): Path to the saved numpy file. Default is 'ground_truth.npy'

    Returns:
        jnp.ndarray: The loaded ground truth array as a JAX array
    """
    try:
        # Load the numpy array
        ground_truth_np = np.load(filepath)
        ground_truth_np = downaverage_states(ground_truth_np, 2)

        # Convert to JAX array for compatibility with your existing code
        ground_truth = jnp.array(ground_truth_np)

        print(f"Shape: {ground_truth.shape}")
        print(f"Data type: {ground_truth.dtype}")

        return ground_truth

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading ground truth: {str(e)}")
        return None


ground_truth = load_ground_truth("data/ground_truth.npy")
if ground_truth is None:
    (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        randomized_variables,
    ) = blast.randomized_initial_blast_state(num_cells_hr)

    config = finalize_config(config, initial_state.shape)

    print("time integrating")
    final_states_hr = time_integration(
        initial_state, config, params, helper_data, registered_variables
    )

    ground_truth = final_states_hr.states

    np.save("data/ground_truth.npy", np.array(ground_truth))

# Training configuration
training_config = TrainingConfig(
    compute_intermediate_losses=True,
    n_look_behind=10,
    loss_function=mse_loss,
    loss_weights=None,
    use_relative_error=False,
    ground_truth_snapshots=ground_truth,
)

# Define neural nets
model = CorrectorCNN(
    in_channels=ground_truth.shape[1],
    hidden_channels=16,
    key=jax.random.PRNGKey(42),
)
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

cnn_mhd_corrector_config = CNNMHDconfig(
    cnn_mhd_corrector=True, network_static=neural_net_static
)

cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)


initial_state, config, params, helper_data, registered_variables, _ = (
    blast.randomized_initial_blast_state(num_cells_hr // downsampling_factor)
)


config = finalize_config(config, initial_state.shape)

config = config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config)
params = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

print("time integrating")

losses, network_params, final_full_sim_data = step_based_training_with_losses(
    initial_state,
    config,
    params,
    helper_data,
    registered_variables,
    training_config,
    ground_truth,
)

print("finished script lol")

plt.figure()
plt.plot(losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("training", "loss_curve.png"))
plt.close()
