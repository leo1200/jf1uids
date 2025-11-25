:py:mod:`jf1uids.time_stepping.time_integration`
================================================

.. py:module:: jf1uids.time_stepping.time_integration

.. autodoc2-docstring:: jf1uids.time_stepping.time_integration
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`time_integration <jf1uids.time_stepping.time_integration.time_integration>`
     - .. autodoc2-docstring:: jf1uids.time_stepping.time_integration.time_integration
          :summary:

API
~~~

.. py:function:: time_integration(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, config: jf1uids.option_classes.simulation_config.SimulationConfig, params: jf1uids.option_classes.simulation_params.SimulationParams, registered_variables: jf1uids.variable_registry.registered_variables.RegisteredVariables, snapshot_callable=None, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None) -> typing.Union[jf1uids.option_classes.simulation_config.STATE_TYPE, jf1uids.data_classes.simulation_snapshot_data.SnapshotData]
   :canonical: jf1uids.time_stepping.time_integration.time_integration

   .. autodoc2-docstring:: jf1uids.time_stepping.time_integration.time_integration
