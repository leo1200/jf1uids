:py:mod:`jf1uids.data_classes.simulation_snapshot_data`
=======================================================

.. py:module:: jf1uids.data_classes.simulation_snapshot_data

.. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SnapshotData <jf1uids.data_classes.simulation_snapshot_data.SnapshotData>`
     - .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData
          :summary:

API
~~~

.. py:class:: SnapshotData
   :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData

   .. py:attribute:: time_points
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.time_points
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.time_points

   .. py:attribute:: states
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.states
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.states

   .. py:attribute:: final_state
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.final_state
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.final_state

   .. py:attribute:: total_mass
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.total_mass
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.total_mass

   .. py:attribute:: total_energy
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.total_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.total_energy

   .. py:attribute:: internal_energy
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.internal_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.internal_energy

   .. py:attribute:: kinetic_energy
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy

   .. py:attribute:: gravitational_energy
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.gravitational_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.gravitational_energy

   .. py:attribute:: radial_momentum
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.radial_momentum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.radial_momentum

   .. py:attribute:: runtime
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.runtime
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.runtime

   .. py:attribute:: num_iterations
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.num_iterations
      :type: int
      :value: 0

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.num_iterations

   .. py:attribute:: current_checkpoint
      :canonical: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.current_checkpoint
      :type: int
      :value: 0

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_snapshot_data.SnapshotData.current_checkpoint
