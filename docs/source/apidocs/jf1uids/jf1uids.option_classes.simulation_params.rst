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
      :value: 0.4

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

   .. py:attribute:: minimum_density
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.minimum_density
      :type: float
      :value: 1e-14

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.minimum_density

   .. py:attribute:: minimum_pressure
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.minimum_pressure
      :type: float
      :value: 1e-14

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.minimum_pressure

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

   .. py:attribute:: snapshot_timepoints
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.snapshot_timepoints
      :type: jax.numpy.array
      :value: 'array(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.snapshot_timepoints

   .. py:attribute:: turbulent_forcing_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.turbulent_forcing_params
      :type: jf1uids._physics_modules._turbulent_forcing._turbulent_forcing_options.TurbulentForcingParams
      :value: 'TurbulentForcingParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.turbulent_forcing_params

   .. py:attribute:: wind_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.wind_params
      :type: jf1uids._physics_modules._stellar_wind.stellar_wind_options.WindParams
      :value: 'WindParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.wind_params

   .. py:attribute:: cosmic_ray_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.cosmic_ray_params
      :type: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams
      :value: 'CosmicRayParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.cosmic_ray_params

   .. py:attribute:: cooling_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.cooling_params
      :type: jf1uids._physics_modules._cooling.cooling_options.CoolingParams
      :value: 'CoolingParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.cooling_params

   .. py:attribute:: neural_net_force_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.neural_net_force_params
      :type: jf1uids._physics_modules._neural_net_force._neural_net_force_options.NeuralNetForceParams
      :value: 'NeuralNetForceParams(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.neural_net_force_params

   .. py:attribute:: cnn_mhd_corrector_params
      :canonical: jf1uids.option_classes.simulation_params.SimulationParams.cnn_mhd_corrector_params
      :type: jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options.CNNMHDconfig
      :value: 'CNNMHDconfig(...)'

      .. autodoc2-docstring:: jf1uids.option_classes.simulation_params.SimulationParams.cnn_mhd_corrector_params
