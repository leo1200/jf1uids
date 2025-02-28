:py:mod:`jf1uids.fluid_equations.registered_variables`
======================================================

.. py:module:: jf1uids.fluid_equations.registered_variables

.. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StaticIntVector <jf1uids.fluid_equations.registered_variables.StaticIntVector>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.StaticIntVector
          :summary:
   * - :py:obj:`RegisteredVariables <jf1uids.fluid_equations.registered_variables.RegisteredVariables>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_registered_variables <jf1uids.fluid_equations.registered_variables.get_registered_variables>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.get_registered_variables
          :summary:

API
~~~

.. py:class:: StaticIntVector
   :canonical: jf1uids.fluid_equations.registered_variables.StaticIntVector

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.StaticIntVector

   .. py:attribute:: x
      :canonical: jf1uids.fluid_equations.registered_variables.StaticIntVector.x
      :type: int
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.StaticIntVector.x

   .. py:attribute:: y
      :canonical: jf1uids.fluid_equations.registered_variables.StaticIntVector.y
      :type: int
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.StaticIntVector.y

   .. py:attribute:: z
      :canonical: jf1uids.fluid_equations.registered_variables.StaticIntVector.z
      :type: int
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.StaticIntVector.z

.. py:class:: RegisteredVariables
   :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables

   .. py:attribute:: num_vars
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.num_vars
      :type: int
      :value: 3

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.num_vars

   .. py:attribute:: density_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.density_index
      :type: int
      :value: 0

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.density_index

   .. py:attribute:: velocity_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.velocity_index
      :type: typing.Union[int, jf1uids.fluid_equations.registered_variables.StaticIntVector]
      :value: 1

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.velocity_index

   .. py:attribute:: magnetic_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.magnetic_index
      :type: typing.Union[int, jf1uids.fluid_equations.registered_variables.StaticIntVector]
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.magnetic_index

   .. py:attribute:: pressure_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.pressure_index
      :type: int
      :value: 2

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.pressure_index

   .. py:attribute:: wind_density_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.wind_density_index
      :type: int
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.wind_density_index

   .. py:attribute:: wind_density_active
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.wind_density_active
      :type: bool
      :value: False

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.wind_density_active

   .. py:attribute:: cosmic_ray_n_index
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.cosmic_ray_n_index
      :type: int
      :value: None

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.cosmic_ray_n_index

   .. py:attribute:: cosmic_ray_n_active
      :canonical: jf1uids.fluid_equations.registered_variables.RegisteredVariables.cosmic_ray_n_active
      :type: bool
      :value: False

      .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.RegisteredVariables.cosmic_ray_n_active

.. py:function:: get_registered_variables(config: jf1uids.option_classes.simulation_config.SimulationConfig) -> jf1uids.fluid_equations.registered_variables.RegisteredVariables
   :canonical: jf1uids.fluid_equations.registered_variables.get_registered_variables

   .. autodoc2-docstring:: jf1uids.fluid_equations.registered_variables.get_registered_variables
