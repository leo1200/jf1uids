from corrector_src.optuna.optuna_train_model import train_model
from hydra import initialize, compose

with initialize(config_path="../../configs", version_base="1.2"):
    cfg = compose(config_name="config", overrides=["data=turbulence"])

initial_loss_calculation_times = [1.0]

epochs = cfg.training.epochs

assert cfg.data.differentiation_mode == 1, "differentiation_mode must be BACKWARDS (1)"

assert not (initial_loss_calculation_times > cfg.data.t_end).any(), (
    "found value greater than the end time in the snapshot timepoints"
)

# ─── Training Configuration ───────────────────────────────────────────

training_config = TrainingConfig()
training_params = TrainingParams(loss_calculation_times=loss_timesteps)

loss_function, compute_loss_from_components, _ = make_loss_function(cfg.training)

# ─── Creating Model And Optimizer ─────────────────────────────────────

model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
model_name = model_cfg.pop("_name_", None)

key = jax.random.PRNGKey(cfg.training.rng_key)
model = instantiate(model_cfg, key=key)

neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
# trainable_params = sum(
#     x.size
#     for x in jax.tree_util.tree_leaves(eqx.filter(neural_net_params, eqx.is_array))
# )
# print(
#     f" ✅ Initialized model '{model_name}' successfully with # of params {trainable_params}"
# )

corrector_config = CorrectorConfig(corrector=True, network_static=neural_net_static)
corrector_params = CorrectorParams(network_params=neural_net_params)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(float(cfg.training.learning_rate)),
)
opt_state = optimizer.init(neural_net_params)

# snapshot_losses = []
epoch_losses = []

gt_cfg_data = cfg.data
# gt_cfg_data.debug = False
dataset_creator = dataset(gt_cfg_data.scenarios, gt_cfg_data)
# print(f" ✅ Using data {dataset_creator.scenario_list}")

# ─── Early Stopping ───────────────────────────────────────────────────

if cfg.training.early_stopping:
    patience = 20
    print(f" 🥱 Using early stopper with patience {patience}")
    early_stopper = EarlyStopper(patience=patience)
    best_params = neural_net_params
else:
    early_stopper = None

if not cfg.data.generate_data_on_fly and len(gt_cfg_data.scenarios) == 1:
    (
        ground_truth,
        sim_bundle_train,
    ) = dataset_creator.train_initializator(
        corrector_config=corrector_config,
        corrector_params=corrector_params,
    )

early_stop = False
for i in range(epochs):
    if cfg.data.generate_data_on_fly:
        (
            ground_truth,
            sim_bundle_train,
        ) = dataset_creator.train_initializator(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )
    else:
        sim_bundle_train.params = sim_bundle_train.params._replace(
            corrector_params=corrector_params
        )
    time_train = time.time()

    losses, new_network_params, opt_state, _ = time_integration_train(
        **sim_bundle_train.unpack_integrate(),
        optimizer=optimizer,
        loss_function=loss_function,
        opt_state=opt_state,
        target_data=ground_truth,
        training_config=training_config,
        training_params=training_params,
    )
    epoch_loss = compute_loss_from_components(losses)
    time_train = time.time() - time_train

    if np.isnan(np.mean(losses)):
        print("nan found in loss, stopping the training")
        raise ValueError("NaN found in loss")

    epoch_losses.append(compute_loss_from_components(losses))

    if early_stopper is not None:
        early_stop = early_stopper.early_stop(epoch_loss)
        if epoch_loss < early_stopper.min_validation_loss:
            best_params = new_network_params
        if early_stop:
            print("🚨 Early stopping 🚨")
            break

    corrector_params = corrector_params._replace(network_params=new_network_params)
    print(
        f" 🟡 Epoch {i} time_train {float(time_train):2f} loss {float(epoch_losses[-1].item()):2f}"
    )

if early_stopper is None:
    best_params = new_network_params
