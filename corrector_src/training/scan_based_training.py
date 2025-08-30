import jax
import jax.numpy as jnp
from typing import Tuple

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union

# jf1uids constants
from jf1uids.option_classes.simulation_config import BACKWARDS, CARTESIAN, FORWARDS, STATE_TYPE

# jf1uids containers
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData

# jf1uids functions
from jf1uids._state_evolution.evolve_state import _evolve_state
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules
from jf1uids.time_stepping._timestep_estimator import _cfl_time_step, _source_term_aware_time_step
from jf1uids.fluid_equations.total_quantities import calculate_internal_energy, calculate_total_mass
from jf1uids.fluid_equations.total_quantities import calculate_total_energy, calculate_kinetic_energy, calculate_gravitational_energy

# timing
from timeit import default_timer as timer

from corrector_src.training.training_config import TrainingConfig
from corrector_src.time_integration_og import _time_integration
def scan_based_training_with_losses(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE
    ) -> Tuple[STATE_TYPE, jnp.ndarray]:
    """
    Use JAX scan to efficiently run simulation with intermediate loss computation.
    
    Returns:
        final_state: Final simulation state
        losses: Array of intermediate losses
    """
    
    # Calculate number of chunks
    total_steps = config.num_checkpoints
    chunk_size = training_config.n_look_behind
    num_chunks = total_steps // chunk_size

    print(total_steps, num_chunks, chunk_size)
    # we must pad the state with ghost cells
    if config.geometry == CARTESIAN:
        # pad the primitive state with two ghost cells on each side
        # to account for the periodic boundary conditions
        original_shape = initial_state.shape

        if config.dimensionality == 1:
            initial_state = jnp.pad(initial_state, ((0, 0), (2, 2)), mode='edge')

        elif config.dimensionality == 2:
            initial_state = jnp.pad(initial_state, ((0, 0), (2, 2), (2, 2)), mode='edge')

        elif config.dimensionality == 3:
            initial_state = jnp.pad(initial_state, ((0, 0), (2, 2), (2, 2), (2, 2)), mode='edge')


    if training_config.return_full_sim:
        full_time_points = jnp.zeros(total_steps)
        full_states = jnp.zeros((total_steps, *original_shape))
        full_total_mass = jnp.zeros(total_steps)
        full_total_energy = jnp.zeros(total_steps)
        full_internal_energy = jnp.zeros(total_steps)
        full_kinetic_energy = jnp.zeros(total_steps)

        if config.self_gravity:
            full_gravitational_energy = jnp.zeros(total_steps)
        else:
            full_gravitational_energy = None

        full_sim_data = SnapshotData(
            time_points=full_time_points,
            states=full_states,
            total_mass=full_total_mass,
            total_energy=full_total_energy,
            internal_energy=full_internal_energy,
            kinetic_energy=full_kinetic_energy,
            gravitational_energy=full_gravitational_energy,
            current_checkpoint=0
        )
    else:
        full_sim_data = None

    def chunk_step(carry, chunk_idx):
        """Process one chunk of the simulation."""
        state, current_time, full_sim_data = carry
        jax.debug.print("chunk_idx = {}", chunk_idx)


        #training_config.current_checkpoint_chunk = 0

        time_points = jnp.zeros(chunk_size)
        states = jnp.zeros((chunk_size, *original_shape))
        total_mass = jnp.zeros(chunk_size)
        total_energy = jnp.zeros(chunk_size)
        internal_energy = jnp.zeros(chunk_size)
        kinetic_energy = jnp.zeros(chunk_size)

        if config.self_gravity:
            gravitational_energy = jnp.zeros(chunk_size)
        else:
            gravitational_energy = None

        current_checkpoint = 0

        chunk_data = SnapshotData(time_points = time_points, states = states, total_mass = total_mass, total_energy = total_energy, internal_energy = internal_energy, kinetic_energy = kinetic_energy, gravitational_energy = gravitational_energy, current_checkpoint = current_checkpoint)

        # Create a mini-loop for this chunk
        def mini_step(chunk_carry, step_idx):
            chunk_state, chunk_time, chunk_data = chunk_carry
            
            # Update chunk data
            def update_snapshot_data(time, state, chunk_data, chunk_i):
                time_points = chunk_data.time_points.at[chunk_i].set(time)

                if config.geometry == CARTESIAN:
                    if config.dimensionality == 1:
                        unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis = 1)
                    elif config.dimensionality == 2:
                        unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis = 1)
                        unpad_state = jax.lax.slice_in_dim(unpad_state, 2, unpad_state.shape[2] - 2, axis = 2)
                    elif config.dimensionality == 3:
                        unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis = 1)
                        unpad_state = jax.lax.slice_in_dim(unpad_state, 2, unpad_state.shape[2] - 2, axis = 2)
                        unpad_state = jax.lax.slice_in_dim(unpad_state, 2, unpad_state.shape[3] - 2, axis = 3)

                states = chunk_data.states.at[chunk_i].set(unpad_state)

                total_mass = chunk_data.total_mass.at[chunk_i].set(calculate_total_mass(unpad_state, helper_data, config))
                total_energy = chunk_data.total_energy.at[chunk_i].set(calculate_total_energy(unpad_state, helper_data, params.gamma, params.gravitational_constant, config, registered_variables))

                internal_energy = chunk_data.internal_energy.at[chunk_i].set(calculate_internal_energy(unpad_state, helper_data, params.gamma, config, registered_variables))
                kinetic_energy = chunk_data.kinetic_energy.at[chunk_i].set(calculate_kinetic_energy(unpad_state, helper_data, config, registered_variables))

                if config.self_gravity:
                    gravitational_energy = chunk_data.gravitational_energy.at[chunk_i].set(calculate_gravitational_energy(unpad_state, helper_data, params.gravitational_constant, config, registered_variables))
                else:
                    gravitational_energy = None

                current_checkpoint = chunk_data.current_checkpoint + 1
                chunk_data = chunk_data._replace(time_points = time_points, states = states, current_checkpoint = current_checkpoint, total_mass = total_mass, total_energy = total_energy, internal_energy = internal_energy, kinetic_energy = kinetic_energy, gravitational_energy = gravitational_energy)
                return chunk_data

            chunk_data = update_snapshot_data(chunk_time, chunk_state, chunk_data, step_idx)

            # Compute dt
            if not config.fixed_timestep:
                if config.source_term_aware_timestep:
                    dt = jax.lax.stop_gradient(_source_term_aware_time_step(
                        chunk_state, config, params, helper_data, registered_variables, chunk_time))
                else:
                    dt = jax.lax.stop_gradient(_cfl_time_step(
                        chunk_state, config.grid_spacing, params.dt_max, params.gamma, 
                        config, registered_variables, params.C_cfl))
            else:
                dt = jnp.asarray(params.t_end / config.num_timesteps)
            
            # Physics and evolution
            chunk_state = _run_physics_modules(
                chunk_state, dt, config, params, helper_data, registered_variables, chunk_time + dt)
            chunk_state = _evolve_state(
                chunk_state, dt, params.gamma, params.gravitational_constant, 
                config, helper_data, registered_variables)
            chunk_time += dt


            return (chunk_state, chunk_time, chunk_data), None
        
        # Run the mini-loop for this chunk

        (final_chunk_state, final_chunk_time, final_chunk_data), _ = jax.lax.scan(
                    mini_step, (state, current_time, chunk_data), jnp.arange(chunk_size)
                )        
        if training_config.return_full_sim:
            # Concatenate chunk data to full simulation data
            def concatenate_to_full_data(full_sim_data, chunk_data, chunk_idx, chunk_size):
                # Dynamic start index along the time axis
                start_idx = chunk_idx * chunk_size
                # If your last chunk can be shorter than chunk_size, you can also do:
                # chunk_len = chunk_data.time_points.shape[0]
                # end_idx = start_idx + chunk_len
                end_idx = start_idx + chunk_size

                def dyn_update(base, update):
                    """
                    Place `update` into `base` starting at (start_idx, 0, 0, ...).
                    Works for 1D and ND arrays where time is axis 0.
                    """
                    starts = (start_idx,) + (0,) * (base.ndim - 1)
                    return jax.lax.dynamic_update_slice(base, update, starts)

                # Update each tracked array along the time axis
                new_time_points     = dyn_update(full_sim_data.time_points,     chunk_data.time_points)
                new_states          = dyn_update(full_sim_data.states,          chunk_data.states)
                new_total_mass      = dyn_update(full_sim_data.total_mass,      chunk_data.total_mass)
                new_total_energy    = dyn_update(full_sim_data.total_energy,    chunk_data.total_energy)
                new_internal_energy = dyn_update(full_sim_data.internal_energy, chunk_data.internal_energy)
                new_kinetic_energy  = dyn_update(full_sim_data.kinetic_energy,  chunk_data.kinetic_energy)

                # Optional gravitational energy (assumes config.self_gravity is a static boolean config)
                if config.self_gravity and (chunk_data.gravitational_energy is not None):
                    new_gravitational_energy = dyn_update(
                        full_sim_data.gravitational_energy,
                        chunk_data.gravitational_energy
                    )
                else:
                    new_gravitational_energy = full_sim_data.gravitational_energy

                # This scalar is fine to carry through the scan state
                new_current_checkpoint = end_idx

                updated_full_sim_data = full_sim_data._replace(
                    time_points=new_time_points,
                    states=new_states,
                    total_mass=new_total_mass,
                    total_energy=new_total_energy,
                    internal_energy=new_internal_energy,
                    kinetic_energy=new_kinetic_energy,
                    gravitational_energy=new_gravitational_energy,
                    current_checkpoint=new_current_checkpoint
                )
                return updated_full_sim_data            
            updated_full_sim_data = concatenate_to_full_data(full_sim_data, final_chunk_data, chunk_idx, chunk_size)
        else:
            updated_full_sim_data = None

        start_idx = chunk_idx * chunk_size
        jax.debug.print("chunk_idx = {}", start_idx)
        ground_truth_chunk = jax.lax.dynamic_slice_in_dim(
            target_data, start_idx, chunk_size, axis=0
        )

        loss = training_config.loss_function(
            chunk_data.states,
            ground_truth_chunk,
            training_config
        )        

        #TODO: ADD NN BACKPROP
        
        return (final_chunk_state, final_chunk_time, updated_full_sim_data), loss
        
    # Run all chunks using scan
    (final_state, final_time, final_full_sim_data), losses = jax.lax.scan(
        chunk_step, (initial_state, 0.0, full_sim_data), jnp.arange(num_chunks)
    )
    
    return final_state, losses, final_full_sim_data