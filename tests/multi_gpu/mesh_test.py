# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 2)
# =======================

import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P

VARAXIS = 0
XAXIS = 1
YAXIS = 2
ZAXIS = 3

@jax.jit
def build_mesh():

    num_cells = 1024

    grid_spacing = 0.1
    ngc = 2
    box_size = 1.0

    split = (1, 1, 1, 2)

    sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    x = jnp.linspace(grid_spacing / 2 - ngc * grid_spacing, box_size - grid_spacing / 2 + ngc * grid_spacing, num_cells + 2 * ngc)
    y = jnp.linspace(grid_spacing / 2 - ngc * grid_spacing, box_size - grid_spacing / 2 + ngc * grid_spacing, num_cells + 2 * ngc)
    z = jnp.linspace(grid_spacing / 2 - ngc * grid_spacing, box_size - grid_spacing / 2 + ngc * grid_spacing, num_cells + 2 * ngc)
    
    # geometric_centers = jax.make_array_from_callback(
    #     global_shape,
    #     named_sharding,
    #     lambda idx: jnp.array(jnp.meshgrid(x[idx[XAXIS]], y[idx[YAXIS]], z[idx[ZAXIS]]))
    # )

    geometric_centers = jax.lax.with_sharding_constraint(jnp.array(jnp.meshgrid(x, y, z)), named_sharding)

    # geometric_centers = jnp.array(jnp.meshgrid(x, y, z))
    # geometric_centers = jax.device_put(geometric_centers, named_sharding)

    return geometric_centers

geometric_centers = build_mesh().block_until_ready()