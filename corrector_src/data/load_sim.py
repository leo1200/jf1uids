import numpy as np
import jax.numpy as jnp
import os

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration
from corrector_src.utils.downaverage import downaverage_states
import corrector_src.data.blast_creation as blast
from typing import Tuple, Any, Optional, List
from omegaconf import DictConfig
from corrector_src.model._corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)


def filepath_state(filepaths: List[str]) -> str:
    """
    Join multiple file path components into a single path.

    Args:
        filepaths (List[str]): A list of path components.

    Returns:
        str: The combined file path.
    """
    return os.path.join(*filepaths)


def load_states(filepath: str = "data/ground_truth"):
    """
    Load the ground truth array from a saved numpy file.
    If created_data_exception is true, data is created on exception and loaded

    Args:
        filepath (str): Path to the saved numpy file. Default is 'ground_truth.npy'
        create_data_exception (bool): if True when an exception is trown the data is created and saved
    Returns:
        jnp.ndarray: The loaded ground truth array as a JAX array
    """
    filepath = filepath
    try:
        ground_truth_np = np.load(filepath)
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


def integrate_blast(
    cfg_data: DictConfig,
    filepath: Optional[str] = None,
    rng_seed: Optional[int] = None,
    downscale: bool = False,
    save_file: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run blast simulation with optional randomization, downscaling, and saving.

    Args:
        cfg_data (DictConfig): Hydra config file with simulation parameters.
        filepath (Optional[str]): Path to save the output (if save_file=True).
        randomized_input_vars (Optional[bool], optional): Randomize input variables. Defaults to None.
        downscale (bool, optional): Whether to downscale the states. Defaults to False.
        save_file (bool, optional): Whether to save results to file. Defaults to True.

    Returns:
        Tuple[np.ndarray, Any]: Final states (HR) and randomized output variables.
    """
    (
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        rng_seed,
    ) = blast.randomized_initial_blast_state(cfg_data.hr_res, cfg_data, rng_seed)

    config = finalize_config(config, initial_state.shape)
    final_states_hr = time_integration(
        initial_state, config, params, helper_data, registered_variables
    )
    states = final_states_hr.states

    if downscale:
        states = downaverage_states(states, cfg_data.downscaling_factor)

    if save_file and filepath is not None:
        np.save(
            filepath,
            np.array(states),
        )
        print(f"{filepath} saved")

    return states, rng_seed


def prepare_initial_state(
    cfg_data: DictConfig,
    rng_seed: Optional[int],
    cnn_mhd_corrector_config: Optional[CNNMHDconfig],
    cnn_mhd_corrector_params: Optional[CNNMHDParams],
    downscale: bool = False,
):
    if downscale:
        resolution = cfg_data.hr_res // cfg_data.downscaling_factor
    else:
        resolution = cfg_data.hr_res
    initial_state, config, params, helper_data, registered_variables, _ = (
        blast.randomized_initial_blast_state(resolution, cfg_data, rng_seed)
    )
    config = finalize_config(config, initial_state.shape)
    if cnn_mhd_corrector_config is not None and cnn_mhd_corrector_params is not None:
        config = config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config)
        params = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)
    return initial_state, config, params, helper_data, registered_variables
