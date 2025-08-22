import jax.numpy as jnp
from corrector_src.training.training_config import TrainingConfig

def mse_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    training_config: TrainingConfig
    ) -> float:
    """
    Default MSE loss function for ground truth training.
    
    Args:
        predicted_state: Current simulation state
        time_step_idx: Current time step index (for indexing ground truth)
        training_config: Training configuration containing ground truth data
    
    Returns:
        MSE loss value
    """
        
    # Compute MSE loss
    if training_config.loss_mask is not None:
        predicted_masked = predicted_state * training_config.loss_mask
        ground_truth_masked = ground_truth * training_config.loss_mask
        loss = jnp.mean(jnp.square(predicted_masked - ground_truth_masked))
    else:
        loss = jnp.mean(jnp.square(predicted_state - ground_truth))
    
    return loss
