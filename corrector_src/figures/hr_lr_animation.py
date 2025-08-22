import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import jax.numpy as jnp

def hr_lr_animate(z_level_hr, z_level_lr, final_states_hr, final_states_lr, gif_name = "blast_hr_lr"):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # Plot 1: Scalar field (e.g., density)
    cax0 = axs[0][0].imshow(
        final_states_hr[0, 0, :, :, z_level_hr].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax0, ax=axs[0][0])
    axs[0][0].set_title("Density")
    axs[0][0].set_xlabel("x")
    axs[0][0].set_ylabel("y")

    # Plot 2: Vector magnitude (e.g., velocity magnitude)
    cax1 = axs[0][1].imshow(
        jnp.sqrt(
            final_states_hr[0, 1, :, :, z_level_hr] ** 2
            + final_states_hr[0, 2, :, :, z_level_hr] ** 2
            + final_states_hr[0, 3, :, :, z_level_hr] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax1, ax=axs[0][1])
    axs[0][1].set_title("Velocity Magnitude")
    axs[0][1].set_xlabel("x")
    axs[0][1].set_ylabel("y")

    # Plot 3: Optional third field, e.g., pressure or similar (reusing animate_vector logic)
    cax2 = axs[0][2].imshow(
        final_states_hr[0, 4, :, :, z_level_hr].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax2, ax=axs[0][2])
    axs[0][2].set_title("Pressure")
    axs[0][2].set_xlabel("x")
    axs[0][2].set_ylabel("y")

    cax3 = axs[0][3].imshow(
        jnp.sqrt(
            final_states_hr[0, 5, :, :, z_level_hr] ** 2
            + final_states_hr[0, 6, :, :, z_level_hr] ** 2
            + final_states_hr[0, 7, :, :, z_level_hr] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    fig.colorbar(cax3, ax=axs[0][3])
    axs[0][3].set_title("Magnetic field")
    axs[0][3].set_xlabel("x")
    axs[0][3].set_ylabel("y")


    # Plot 1: Scalar field (e.g., density)
    cax_lr_0 = axs[1][0].imshow(
        final_states_lr[0, 0, :, :, z_level_lr].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax_lr_0, ax=axs[1][0])
    axs[1][0].set_title("Density")
    axs[1][0].set_xlabel("x")
    axs[1][0].set_ylabel("y")

    # Plot 2: Vector magnitude (e.g., velocity magnitude)
    cax_lr_1 = axs[1][1].imshow(
        jnp.sqrt(
            final_states_lr[0, 1, :, :, z_level_lr] ** 2
            + final_states_lr[0, 2, :, :, z_level_lr] ** 2
            + final_states_lr[0, 3, :, :, z_level_lr] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax_lr_1, ax=axs[1][1])
    axs[1][1].set_title("Velocity Magnitude")
    axs[1][1].set_xlabel("x")
    axs[1][1].set_ylabel("y")

    # Plot 3: Optional third field, e.g., pressure or similar (reusing animate_vector logic)
    cax_lr_2 = axs[1][2].imshow(
        final_states_lr[0, 4, :, :, z_level_lr].T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    fig.colorbar(cax_lr_2, ax=axs[1][2])
    axs[1][2].set_title("Pressure")
    axs[1][2].set_xlabel("x")
    axs[1][2].set_ylabel("y")

    cax_lr_3 = axs[1][3].imshow(
        jnp.sqrt(
            final_states_lr[0, 5, :, :, z_level_lr] ** 2
            + final_states_lr[0, 6, :, :, z_level_lr] ** 2
            + final_states_lr[0, 7, :, :, z_level_lr] ** 2
        ).T,
        origin="lower",
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    fig.colorbar(cax_lr_3, ax=axs[1][3])
    axs[1][3].set_title("Magnetic field")
    axs[1][3].set_xlabel("x")
    axs[1][3].set_ylabel("y")

    # Update function for all three plots
    def animate_all(i):
        cax0.set_array(final_states_hr[i, 0, :, :, z_level_hr].T)
        cax1.set_array(
            jnp.sqrt(
                final_states_hr[i, 1, :, :, z_level_hr] ** 2
                + final_states_hr[i, 2, :, :, z_level_hr] ** 2
                + final_states_hr[i, 3, :, :, z_level_hr] ** 2
            ).T
        )
        cax2.set_array(final_states_hr[i, 4, :, :, z_level_hr].T)
        cax3.set_array(
            jnp.sqrt(
                final_states_hr[i, 5, :, :, z_level_hr] ** 2
                + final_states_hr[i, 6, :, :, z_level_hr] ** 2
                + final_states_hr[i, 7, :, :, z_level_hr] ** 2
            ).T
        )
        cax_lr_0.set_array(final_states_lr[i, 0, :, :, z_level_lr].T)
        cax_lr_1.set_array(
            jnp.sqrt(
                final_states_lr[i, 1, :, :, z_level_lr] ** 2
                + final_states_lr[i, 2, :, :, z_level_lr] ** 2
                + final_states_lr[i, 3, :, :, z_level_lr] ** 2
            ).T
        )
        cax_lr_2.set_array(final_states_lr[i, 4, :, :, z_level_lr].T)
        cax_lr_3.set_array(
            jnp.sqrt(
                final_states_lr[i, 5, :, :, z_level_lr] ** 2
                + final_states_lr[i, 6, :, :, z_level_lr] ** 2
                + final_states_lr[i, 7, :, :, z_level_lr] ** 2
            ).T
        )

        return cax0, cax1, cax2, cax3, cax_lr_0, cax_lr_1, cax_lr_2, cax_lr_3

    ani = animation.FuncAnimation(fig, animate_all, frames=final_states_hr.shape[0], interval=50)

    ani.save("figures/" + gif_name + ".gif")
    plt.show()



