import jax.numpy as jnp
from corrector_src.training.training_config import TrainingConfig


def mse_loss(predicted_state: jnp.ndarray, ground_truth: jnp.ndarray) -> float:
    """
    Default MSE loss function for ground truth training.

    Args:
        predicted_state: Current simulation state
        ground_truth:
        training_config: Training configuration containing ground truth data

    Returns:
        MSE loss value
    """

    # Compute MSE loss
    loss = jnp.mean(jnp.square(predicted_state - ground_truth))

    return loss
