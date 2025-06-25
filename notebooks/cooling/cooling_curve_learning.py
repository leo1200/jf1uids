import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np # For initial data

# -----------------------------------------------------------------------------
# Define invertible blocks and network
# -----------------------------------------------------------------------------

class AffineSoftplus1D(eqx.Module): # Changed from AffineLeaky1D
    s: jax.Array  # log scale (w = exp(s))
    b: jax.Array  # bias
    # Removed: a: jax.Array  # log slope for LeakyReLU

    def __init__(self, key: jax.random.PRNGKey): # key is kept for consistency, though not used for s, b here
        # Parameters are initialized to specific values (zeros).
        self.s = jnp.zeros(()) # log_scale = 0 => scale = 1
        self.b = jnp.zeros(()) # bias = 0
        # Parameter 'a' for LeakyReLU is removed.

    def __call__(self, x: jax.Array) -> jax.Array:
        w = jnp.exp(self.s)
        # self.b is used directly in the next line
        affine_transform_x = w * x + self.b
        # Apply softplus activation instead of LeakyReLU
        # jax.nn.softplus(x) = log(1 + exp(x))
        z = jax.nn.softplus(affine_transform_x)
        return z

    def inverse(self, z: jax.Array) -> jax.Array:
        # Inverse of softplus: y = log(exp(z) - 1)
        # This is defined for z > 0. The output of jax.nn.softplus is always > 0.
        # jnp.expm1(z) computes exp(z) - 1 accurately.
        # We take jnp.log of that result.
        deactivated_z = jnp.log(jnp.expm1(z))
        
        w = jnp.exp(self.s)
        # self.b is used directly in the next line
        # Inverse of affine transformation
        # w = exp(s) is always > 0, so no division by zero unless s is -inf (highly unlikely).
        x = (deactivated_z - self.b) / w
        return x


class Invertible1DNet(eqx.Module):
    layers: list

    def __init__(self, num_blocks: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, num_blocks)
        # Use the new AffineSoftplus1D block
        self.layers = [AffineSoftplus1D(k) for k in keys] # Changed class here

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.log(x) # Log-scale input
        for layer in self.layers:
            x = layer(x)
        x = jnp.exp(x) # Final exponentiation
        return x

    def inverse(self, z: jax.Array) -> jax.Array:
        z = jnp.log(z) # Undo exponentiation; requires z > 0
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        z = jnp.exp(z) # Undo log-scale
        return z

# -----------------------------------------------------------------------------
# Loss and Training Step
# (This section remains unchanged)
# -----------------------------------------------------------------------------

def loss_fn(
    net: Invertible1DNet,
    T_norm_batch: jax.Array,    # Normalized T values
    N_target_batch: jax.Array,  # Target N values
    std_T_val: jax.Array        # Standard deviation of original T values (for chain rule)
) -> jax.Array:
    grad_net_wrt_T_norm_fn = jax.grad(net)
    N_pred_times_std_T = jax.vmap(grad_net_wrt_T_norm_fn)(T_norm_batch)
    N_predicted = N_pred_times_std_T / std_T_val
    
    log_N_predicted = jnp.log(N_predicted)
    log_N_target = jnp.log(N_target_batch)

    return jnp.mean((log_N_target - log_N_predicted)**2)

@eqx.filter_jit
def make_step(
    net: Invertible1DNet,
    opt_state: optax.OptState,
    T_norm_batch: jax.Array,
    N_target_batch: jax.Array,
    std_T_val: jax.Array,
    optimizer: optax.GradientTransformation
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
        net, T_norm_batch, N_target_batch, std_T_val
    )
    updates, opt_state = optimizer.update(grads, opt_state, net)
    net = eqx.apply_updates(net, updates)
    return net, opt_state, loss_value

# -----------------------------------------------------------------------------
# Training loop
# (This section remains unchanged)
# -----------------------------------------------------------------------------

def train(
    net_init: Invertible1DNet,
    T_data: jax.Array,          # Original T values
    N_data: jax.Array,          # Original N values
    num_epochs: int,
    lr: float,
    key: jax.random.PRNGKey
):
    mean_T = jnp.mean(T_data)
    std_T = jnp.std(T_data)
    
    if std_T == 0.0:
        print("Warning: Standard deviation of T_data is zero. Using std_T = 1.0 for normalization.")
        std_T = jnp.array(1.0)
        
    T_norm_train = T_data / std_T
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(net_init, eqx.is_array))
    
    net = net_init
    losses = []
    print_interval = max(1, num_epochs // 20)

    for epoch in range(1, num_epochs + 1):
        net, opt_state, loss_val = make_step(
            net, opt_state, T_norm_train, N_data, std_T, optimizer
        )
        losses.append(loss_val.item())
        
        if epoch == 1 or epoch % print_interval == 0 or epoch == num_epochs:
            print(f"Epoch {epoch}/{num_epochs}   Loss = {loss_val.item():.6e}")
            
    return net, losses, mean_T, std_T

# -----------------------------------------------------------------------------
# Data
# (This section remains unchanged)
# -----------------------------------------------------------------------------
log10_T_np = np.array([
    3.80, 3.84, 3.88, 3.92, 3.96, 4.00, 4.04, 4.08, 4.12, 4.16,
    4.20, 4.24, 4.28, 4.32, 4.36, 4.40, 4.44, 4.48, 4.52, 4.56,
    4.60, 4.64, 4.68, 4.72, 4.76, 4.80, 4.84, 4.88, 4.92, 4.96,
    5.00, 5.04, 5.08, 5.12, 5.16, 5.20, 5.24, 5.28, 5.32, 5.36,
    5.40, 5.44, 5.48, 5.52, 5.56, 5.60, 5.64, 5.68, 5.72, 5.76,
    5.80, 5.84, 5.88, 5.92, 5.96, 6.00, 6.04, 6.08, 6.12, 6.16,
    6.20, 6.24, 6.28, 6.32, 6.36, 6.40, 6.44, 6.48, 6.52, 6.56,
    6.60, 6.64, 6.68, 6.72, 6.76, 6.80, 6.84, 6.88, 6.92, 6.96,
    7.00, 7.04, 7.08, 7.12, 7.16, 7.20, 7.24, 7.28, 7.32, 7.36,
    7.40, 7.44, 7.48, 7.52, 7.56, 7.60, 7.64, 7.68, 7.72, 7.76,
    7.80, 7.84, 7.88, 7.92, 7.96, 8.00, 8.04, 8.08, 8.12, 8.16
])

log10_Lambda_N_np = np.array([
    -25.7331, -25.0383, -24.4059, -23.8288, -23.3027, -22.8242, -22.3917, -22.0067, -21.6818, -21.4529,
    -21.3246, -21.3459, -21.4305, -21.5293, -21.6138, -21.6615, -21.6551, -21.5919, -21.5092, -21.4124,
    -21.3085, -21.2047, -21.1067, -21.0194, -20.9413, -20.8735, -20.8205, -20.7805, -20.7547, -20.7455,
    -20.7565, -20.7820, -20.8008, -20.7994, -20.7847, -20.7687, -20.7590, -20.7544, -20.7505, -20.7545,
    -20.7888, -20.8832, -21.0450, -21.2286, -21.3737, -21.4573, -21.4935, -21.5098, -21.5345, -21.5863,
    -21.6548, -21.7108, -21.7424, -21.7576, -21.7696, -21.7883, -21.8115, -21.8303, -21.8419, -21.8514,
    -21.8690, -21.9057, -21.9690, -22.0554, -22.1488, -22.2355, -22.3084, -22.3641, -22.4033, -22.4282,
    -22.4408, -22.4443, -22.4411, -22.4334, -22.4242, -22.4164, -22.4134, -22.4168, -22.4267, -22.4418,
    -22.4603, -22.4830, -22.5112, -22.5449, -22.5819, -22.6177, -22.6483, -22.6719, -22.6883, -22.6985,
    -22.7032, -22.7037, -22.7008, -22.6950, -22.6869, -22.6769, -22.6655, -22.6531, -22.6397, -22.6258,
    -22.6111, -22.5964, -22.5816, -22.5668, -22.5519, -22.5367, -22.5216, -22.5062, -22.4912, -22.4753
])

log10_T_jax = jnp.array(log10_T_np)
log10_Lambda_N_jax = jnp.array(log10_Lambda_N_np)

T_train_data = jnp.power(10.0, log10_T_jax)
# The print(jnp.log(T_train_data)) was for debugging, can be removed or kept
# print(jnp.log(T_train_data)) 
N_train_data = jnp.power(10.0, log10_Lambda_N_jax)

# -----------------------------------------------------------------------------
# Main script execution
# (This section remains unchanged)
# -----------------------------------------------------------------------------

def main():
    NUM_BLOCKS = 100
    LEARNING_RATE = 4e-4 
    NUM_EPOCHS = 120000   
    SEED = 42

    key = jax.random.PRNGKey(SEED)
    model_key, train_key = jax.random.split(key)
    net_initial = Invertible1DNet(num_blocks=NUM_BLOCKS, key=model_key)

    print("Starting training...")
    trained_net, losses, mean_T, std_T = train(
        net_initial, T_train_data, N_train_data, NUM_EPOCHS, LEARNING_RATE, train_key
    )
    print("Training finished.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE on ln N scale)")
    plt.title("Training Loss Curve")
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.tight_layout()
    plt.show()

    T_values_for_plot = jnp.sort(T_train_data)
    T_norm_for_plot = T_values_for_plot / std_T
    
    N_pred_times_std_T_plot = jax.vmap(jax.grad(trained_net))(T_norm_for_plot)
    N_predicted_plot_values = N_pred_times_std_T_plot / std_T

    plt.figure(figsize=(10, 6))
    plt.scatter(T_train_data, N_train_data, label="True Data (N vs T)", s=20, color='blue', alpha=0.6, zorder=2)
    plt.plot(T_values_for_plot, N_predicted_plot_values, label="Model Prediction (N vs T)", color='red', linewidth=2, zorder=3)
    
    plt.xlabel("T (Temperature)")
    plt.ylabel("N (Cooling Rate $\Lambda_N$)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Fit of Cooling Function $\Lambda_N(T)$ (Log-Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

    log10_T_plot = jnp.log10(T_values_for_plot)
    log10_N_predicted_plot = jnp.log10(N_predicted_plot_values)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(log10_T_jax, log10_Lambda_N_jax, label="True Data ($log_{10} \Lambda_N$ vs $log_{10} T$)", s=20, color='blue', alpha=0.6, zorder=2)
    plt.plot(log10_T_plot, log10_N_predicted_plot, label="Model Prediction ($log_{10} \Lambda_N$ vs $log_{10} T$)", color='red', linewidth=2, zorder=3)
    
    plt.xlabel("$log_{10}(T)$")
    plt.ylabel("$log_{10}(\Lambda_N)$")
    plt.title("Fit of Cooling Function $\Lambda_N(T)$ (Base-10 Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig("cooling_curve_fit.svg")

if __name__ == '__main__':
    main()