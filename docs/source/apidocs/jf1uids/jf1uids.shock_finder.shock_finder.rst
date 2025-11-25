:py:mod:`jf1uids.shock_finder.shock_finder`
===========================================

.. py:module:: jf1uids.shock_finder.shock_finder

.. autodoc2-docstring:: jf1uids.shock_finder.shock_finder
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`shock_sensor <jf1uids.shock_finder.shock_finder.shock_sensor>`
     - .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.shock_sensor
          :summary:
   * - :py:obj:`shock_criteria <jf1uids.shock_finder.shock_finder.shock_criteria>`
     - .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.shock_criteria
          :summary:
   * - :py:obj:`find_shock_zone <jf1uids.shock_finder.shock_finder.find_shock_zone>`
     - .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.find_shock_zone
          :summary:

API
~~~

.. py:function:: shock_sensor(pressure: jf1uids.option_classes.simulation_config.FIELD_TYPE) -> jf1uids.option_classes.simulation_config.FIELD_TYPE
   :canonical: jf1uids.shock_finder.shock_finder.shock_sensor

   .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.shock_sensor

.. py:function:: shock_criteria(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.variable_registry.registered_variables.RegisteredVariables, helper_data: jf1uids.data_classes.simulation_helper_data.HelperData) -> jax.numpy.ndarray
   :canonical: jf1uids.shock_finder.shock_finder.shock_criteria

   .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.shock_criteria

.. py:function:: find_shock_zone(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.variable_registry.registered_variables.RegisteredVariables, helper_data: jf1uids.data_classes.simulation_helper_data.HelperData) -> typing.Tuple[typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]], typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]], typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]]]
   :canonical: jf1uids.shock_finder.shock_finder.find_shock_zone

   .. autodoc2-docstring:: jf1uids.shock_finder.shock_finder.find_shock_zone
