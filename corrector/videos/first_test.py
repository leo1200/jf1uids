from autocvd import autocvd
autocvd(num_gpus=1)
import os
import numpy as np
import pyvista as pv
# ---------------------------------------------------------------------
# Load MHD states
# ---------------------------------------------------------------------
filepath = "/export/home/jalegria/Thesis/jf1uids/corrector_src/data/states/ground_truth_100.npy"
radius = 80.0
frames_per_state = 5

states = np.load(filepath)
n_states = states.shape[0]
n_frames = frames_per_state * n_states

# Assume shape: (time, channel, nx, ny, nz)
rho = states[0, 0]
n = rho.shape[-1]
print(n)
vmin, vmax = float(states[-1, 0].min()), float(states[-1, 0].max())
print(f"Density range: {vmin:.3e} – {vmax:.3e}")

# Create base grid
grid = pv.ImageData()
grid.dimensions = np.array(rho.shape)
grid.spacing = (1.0, 1.0, 1.0)
grid.origin = (0, 0, 0)
grid["density"] = rho.flatten(order="F")

# ---------------------------------------------------------------------
# Plotter: two subplots (1 row × 2 columns)
# ---------------------------------------------------------------------
plotter = pv.Plotter(shape=(1, 2), off_screen=True)

# Left subplot: volume rendering
plotter.subplot(0, 0)
plotter.add_text("3D Volume View", font_size=14)
# opacity = [0.0, 0.0, 0.0, 0.1, 0.3, 0.8, 1.0]

# Create a smooth opacity transfer function with 256 steps
opacity_values = [0, 0.0, 0.0, 0.02, 0.15, 0.5, 0.7, 0.9, 1.0]
  # gamma correction for smoother look
# Optional: reduce max opacity so you can see inside the cloud

vol_actor = plotter.add_volume(grid, scalars="density", cmap="viridis", opacity=opacity_values, clim=[vmin, vmax])

# Right subplot: orthogonal slices
plotter.subplot(0, 1)
plotter.add_text("Orthogonal Slices", font_size=14)
slices_actor = plotter.add_mesh(grid.slice_orthogonal(), cmap="viridis")
plotter.add_axes()

camera_dot = pv.Sphere(radius=1.0, center=(0, 0, 0))
dot_actor = plotter.add_mesh(camera_dot, color='red')

# ---------------------------------------------------------------------
# Camera setup (orbit for left view)
# ---------------------------------------------------------------------

center = np.array([n / 2, n / 2, n / 2])
angles = np.linspace(0, np.pi, n_frames)
camera_positions = np.array([
    [center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), n / 2]
    for a in angles
])
focal_point = tuple(center)
viewup = (0, 0, 1)

# ---------------------------------------------------------------------
# Output video
# ---------------------------------------------------------------------
out_name = "corrector/videos/mhd_evolution.mp4"
os.makedirs(os.path.dirname(out_name), exist_ok=True)
if os.path.exists(out_name):
    os.remove(out_name)

plotter.open_movie(out_name, framerate=30)
plotter.subplot(0, 0)
plotter.camera_position = [tuple(camera_positions[0]), focal_point, viewup]
plotter.render()

# ---------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------
for frame_i, cam_pos in enumerate(camera_positions):
    state_idx = frame_i // frames_per_state
    rho = states[state_idx, 0]
    grid.point_data["density"] = rho.ravel(order="F")
    
    # Update volume rendering on left
    plotter.subplot(0, 0)
    plotter.remove_actor(vol_actor)
    vol_actor = plotter.add_volume(
        grid, 
        scalars="density", 
        cmap="viridis", 
        opacity=opacity_values,
        clim=[vmin, vmax]
    )
    plotter.camera_position = [tuple(cam_pos), focal_point, viewup]
    
    # Update slices on right
    plotter.subplot(0, 1)
    plotter.remove_actor(slices_actor)
    slices_actor = plotter.add_mesh(grid.slice_orthogonal(), cmap="viridis", clim=[vmin, vmax])
    
    # Update red dot position (camera marker)
    plotter.remove_actor(dot_actor)
    camera_dot = pv.Sphere(radius=1.0, center=tuple(cam_pos))
    dot_actor = plotter.add_mesh(camera_dot, color='red')
    
    plotter.render()
    plotter.write_frame()
plotter.close()
print(f"✅ Saved movie to: {out_name}")
