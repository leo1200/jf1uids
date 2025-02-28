:py:mod:`jf1uids.option_classes.simulation_params`
==================================================

.. py:module:: jf1uids.option_classes.simulation_params

.. autodoc2-docstring:: jf1uids.option_classes.simulation_params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SimulationParams <jf1uids.option_classes.simulation_params.SimulationParams>`
     - .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams
          :summary:

API
~~~

.. py:class:: SimulationParams
   :canonical: jf1uids.option_classes.simulation_params.SimulationParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams

   .. py:attribute:: C_cfl
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.C_cfl
      :type: float
      :value: 0.8

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.C_cfl

   .. py:attribute:: gravitational_constant
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.gravitational_constant
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.gravitational_constant

   .. py:attribute:: gamma
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.gamma
      :type: float
      :value: None

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.gamma

   .. py:attribute:: dt_max
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.dt_max
      :type: float
      :value: 0.001

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.dt_max

   .. py:attribute:: t_end
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.t_end
      :type: float
      :value: 0.2

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.t_end

   .. py:attribute:: wind_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.wind_params
      :type: jf1uids._physics_modules._stellar_wind.stellar_wind_options.WindParams
      :value: 'WindParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.wind_params
