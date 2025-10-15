:py:mod:`jf1uids._physics_modules._cosmic_rays.cosmic_ray_options`
==================================================================

.. py:module:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options

.. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CosmicRayConfig <jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig
          :summary:
   * - :py:obj:`CosmicRayParams <jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams
          :summary:

API
~~~

.. py:class:: CosmicRayConfig
   :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig

   .. py:attribute:: cosmic_rays
      :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig.cosmic_rays
      :type: bool
      :value: False

      .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig.cosmic_rays

   .. py:attribute:: diffusive_shock_acceleration
      :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig.diffusive_shock_acceleration
      :type: bool
      :value: False

      .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayConfig.diffusive_shock_acceleration

.. py:class:: CosmicRayParams
   :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams

   .. py:attribute:: diffusive_shock_acceleration_start_time
      :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams.diffusive_shock_acceleration_start_time
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams.diffusive_shock_acceleration_start_time

   .. py:attribute:: diffusive_shock_acceleration_efficiency
      :canonical: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams.diffusive_shock_acceleration_efficiency
      :type: float
      :value: 0.1

      .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cosmic_ray_options.CosmicRayParams.diffusive_shock_acceleration_efficiency
