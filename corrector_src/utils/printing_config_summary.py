def print_data_config_summary(cfg_data):
    """Prints a formatted and emoji-stylized summary of the Hydra data configuration."""
    # Define fields to display (skip the unwanted ones)
    fields = {
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

    # Helper to stylize booleans
    def fmt_value(val):
        if isinstance(val, bool):
            return "✅" if val else "❌"
        return str(val)

    # Collect available fields and formatted values
    info = []
    for key, label in fields.items():
        if hasattr(cfg_data, key):
            val = getattr(cfg_data, key)
            info.append((label, fmt_value(val)))

    # Compute alignment width
    label_width = max(len(label) for label, _ in info) + 2

    # Print header
    print("\n" + "=" * 60)
    print("📦  Hydra Data Configuration Summary")
    print("=" * 60)

    # Print each field aligned
    for label, val in info:
        print(f"{label:<{label_width}} : {val}")

    print("=" * 60 + "\n")
