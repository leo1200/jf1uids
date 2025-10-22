import jax
import jax.numpy as jnp
from jf1uids.fluid_equations.fluid import (
    get_absolute_velocity,
    total_energy_from_primitives,
)
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)
from jf1uids.option_classes.simulation_params import SimulationParams
import Pk_library as PKL
from jf1uids._physics_modules._mhd._vector_maths import divergence3D


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


def get_energy(
    primitive_state: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
):
    """
    Calculate the total energy from the primitive state. The resolution doesnt matter, only the dimensionality
    """
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    return total_energy_from_primitives(rho, u, p, params.gamma)


def spectral_energy_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> float:
    """
    Important caveat: predicted and ground truth must be same size, also same box size
    """
    use_log_ratio = True
    energy_predicted = get_energy(predicted_state, config, registered_variables, params)
    energy_gt = get_energy(ground_truth, config, registered_variables, params)

    pk_pred = PKL.Pk(
        delta=energy_predicted, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
    )

    pk_gt = PKL.Pk(
        delta=energy_gt, BoxSize=1, axis=0, MAS="None", threads=6, verbose=False
    )
    if use_log_ratio:
        # Extract 1D spectra and handle possible numerical issues
        P_pred = jnp.array(pk_pred.Pk[:, 0])
        P_true = jnp.array(pk_gt.Pk[:, 0])

        # Avoid division by zero or log of 0
        eps = 1e-12
        P_pred = jnp.clip(P_pred, eps, None)
        P_true = jnp.clip(P_true, eps, None)

        # Compute spectral log-ratio loss
        log_ratio = jnp.log(P_pred / P_true)
        loss = jnp.sqrt(jnp.mean(log_ratio**2))
    else:
        loss = jnp.mean((pk_pred.Pk[:, 0] - pk_gt.Pk[:, 0]) ** 2)

    return loss


def rate_of_strain(
    state: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    velocity = state[registered_variables.velocity_index, ...]
    dx = config.grid_spacing
    # Compute ∂v_i/∂x_j using jnp.gradient
    dvx_dx, dvx_dy, dvx_dz = jnp.gradient(velocity[0], dx)
    dvy_dx, dvy_dy, dvy_dz = jnp.gradient(velocity[1], dx)
    dvz_dx, dvz_dy, dvz_dz = jnp.gradient(velocity[2], dx)

    grad_v = jnp.stack(
        [
            jnp.stack([dvx_dx, dvx_dy, dvx_dz]),
            jnp.stack([dvy_dx, dvy_dy, dvy_dz]),
            jnp.stack([dvz_dx, dvz_dy, dvz_dz]),
        ]
    )  # shape (3, 3, Nx, Ny, Nz)
    rate_of_strain = 0.5 * (grad_v + jnp.swapaxes(grad_v, 0, 1))
    return rate_of_strain


def rate_of_strain_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> float:
    strain_predicted = rate_of_strain(
        state=predicted_state, config=config, registered_variables=registered_variables
    )
    strain_gt = rate_of_strain(
        state=ground_truth, config=config, registered_variables=registered_variables
    )
    loss = mse_loss(strain_predicted, strain_gt)
    return loss


# def mean_flow_loss() need to think this one trough as the average over time, i think it should be covered by the rolling
# if i wanted to implement it, id have to play with the time integration fs
