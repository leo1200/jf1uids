:orphan:

:py:mod:`jf1uids.time_stepping._timestep_estimator`
===================================================

.. py:module:: jf1uids.time_stepping._timestep_estimator

.. autodoc2-docstring:: jf1uids.time_stepping._timestep_estimator
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_wave_speeds <jf1uids.time_stepping._timestep_estimator.get_wave_speeds>`
     - .. autodoc2-docstring:: jf1uids.time_stepping._timestep_estimator.get_wave_speeds
          :summary:

API
~~~

.. py:function:: get_wave_speeds(primitives_left: jf1uids.option_classes.simulation_config.STATE_TYPE, primitives_right: jf1uids.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables, config: jf1uids.option_classes.simulation_config.SimulationConfig, flux_direction_index: int) -> typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]]
   :canonical: jf1uids.time_stepping._timestep_estimator.get_wave_speeds

   .. autodoc2-docstring:: jf1uids.time_stepping._timestep_estimator.get_wave_speeds
