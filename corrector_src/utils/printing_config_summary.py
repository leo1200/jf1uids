def print_full_config_summary(cfg_data, cfg_training=None, cfg_model=None):
    """Prints formatted and emoji-stylized summaries of the Hydra data, training, and model configurations."""

    def fmt_value(val):
        """Stylize booleans and convert others to strings."""
        if isinstance(val, bool):
            return "✅" if val else "❌"
        return str(val)

    def print_section(title, fields, cfg):
        """Helper to print a formatted section for any config."""
        if cfg is None:
            return
        info = []
        for key, label in fields.items():
            val = getattr(cfg, key, None)
            if val is not None:
                info.append((label, fmt_value(val)))
        if not info:
            return
        label_width = max(len(label) for label, _ in info) + 2
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        for label, val in info:
            print(f"{label:<{label_width}} : {val}")
        print("=" * 60)

    # ---------------- Data Config ---------------- #
    data_fields = {
        "hr_res": "🧩  High Resolution Size",
        "downscaling_factor": "🔽  Downscaling Factor",
        "num_checkpoints": "💾  Number of Checkpoints",
        "num_timesteps": "⏱️  Number of Timesteps",
        "generate_data_on_fly": "⚙️  Generate Data On-the-Fly",
        "precomputed_data": "📁  Use Precomputed Data",
        "fixed_timestep": "📉  Fixed Timestep",
        "snapshot_timepoints": "🕐  Snapshot Timepoints",
        "use_specific_snapshot_timepoints": "🎯  Use Specific Timepoints",
    }
    print_section("📦  Hydra Data Configuration Summary", data_fields, cfg_data)

    # ---------------- Training Config ---------------- #
    train_fields = {
        "epochs": "🏋️  Epochs",
        "n_look_behind": "👀  Look Behind Steps",
        "learning_rate": "⚡  Learning Rate",
        "return_full_sim": "🌊  Return Full Simulation",
        "return_full_sim_epoch_interval": "🕓  Full Sim Epoch Interval",
        "rng_key": "🎲  RNG Key",
        "debug": "🐞  Debug Mode",
        "mse_loss": "📏  MSE Loss Weight",
        "spectral_energy_loss": "🌈  Spectral Energy Loss Weight",
        "rate_of_strain_loss": "💨  Rate-of-Strain Loss Weight",
        "early_stopping": "🛑  Early Stopping",
        "patience": "⌛  Patience (Epochs)",
        "correct_from_beggining": "🎯  Correct from Beginning",
        "delayed_correction_time": "⏳  Delayed Correction Time",
    }
    print_section("🧠  Training Configuration Summary", train_fields, cfg_training)

    # ---------------- Model Config ---------------- #
    if cfg_model is not None:
        model_type = getattr(cfg_model, "_name_", getattr(cfg_model, "__name__", "unknown")).lower()
        if "fno" in model_type:
            model_fields = {
                "_target_": "🎯  Model Target",
                "_name_": "🏷️  Model Name",
                "hidden_channels": "🔒  Hidden Channels",
                "n_fourier_layers": "🌀  Fourier Layers",
                "fourier_modes": "🌐  Fourier Modes",
                "shifting_modes": "↔️  Shifting Modes",
                "postprocessing_floor": "🧱  Postprocessing Floor",
                "output_channels": "📤  Output Channels",
            }
        elif "cnn" in model_type:
            model_fields = {
                "_target_": "🎯  Model Target",
                "__name__": "🏷️  Model Name",
                "in_channels": "📥  Input Channels",
                "hidden_channels": "🔒  Hidden Channels",
            }
        else:
            model_fields = {
                "_target_": "🎯  Model Target",
                "_name_": "🏷️  Model Name",
            }
        print_section("🧩  Model Configuration Summary", model_fields, cfg_model)
