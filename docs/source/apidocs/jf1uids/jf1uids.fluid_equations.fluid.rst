:py:mod:`jf1uids.fluid_equations.fluid`
=======================================

.. py:module:: jf1uids.fluid_equations.fluid

.. autodoc2-docstring:: jf1uids.fluid_equations.fluid
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`primitive_state_from_conserved <jf1uids.fluid_equations.fluid.primitive_state_from_conserved>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.primitive_state_from_conserved
          :summary:
   * - :py:obj:`conserved_state_from_primitive <jf1uids.fluid_equations.fluid.conserved_state_from_primitive>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.conserved_state_from_primitive
          :summary:
   * - :py:obj:`get_absolute_velocity <jf1uids.fluid_equations.fluid.get_absolute_velocity>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.get_absolute_velocity
          :summary:
   * - :py:obj:`pressure_from_internal_energy <jf1uids.fluid_equations.fluid.pressure_from_internal_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.pressure_from_internal_energy
          :summary:
   * - :py:obj:`internal_energy_from_energy <jf1uids.fluid_equations.fluid.internal_energy_from_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.internal_energy_from_energy
          :summary:
   * - :py:obj:`pressure_from_energy <jf1uids.fluid_equations.fluid.pressure_from_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.pressure_from_energy
          :summary:
   * - :py:obj:`total_energy_from_primitives <jf1uids.fluid_equations.fluid.total_energy_from_primitives>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.total_energy_from_primitives
          :summary:
   * - :py:obj:`speed_of_sound <jf1uids.fluid_equations.fluid.speed_of_sound>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.speed_of_sound
          :summary:

API
~~~

.. py:function:: primitive_state_from_conserved(conserved_state: jf1uids.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jf1uids.option_classes.simulation_config.STATE_TYPE
   :canonical: jf1uids.fluid_equations.fluid.primitive_state_from_conserved

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.primitive_state_from_conserved

.. py:function:: conserved_state_from_primitive(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jf1uids.option_classes.simulation_config.STATE_TYPE
   :canonical: jf1uids.fluid_equations.fluid.conserved_state_from_primitive

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.conserved_state_from_primitive

.. py:function:: get_absolute_velocity(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> typing.Union[jaxtyping.Float[jaxtyping.Array, num_cells], jaxtyping.Float[jaxtyping.Array, num_cells_x num_cells_y], jaxtyping.Float[jaxtyping.Array, num_cells_x num_cells_y num_cells_z]]
   :canonical: jf1uids.fluid_equations.fluid.get_absolute_velocity

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.get_absolute_velocity

.. py:function:: pressure_from_internal_energy(e, rho, gamma)
   :canonical: jf1uids.fluid_equations.fluid.pressure_from_internal_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.pressure_from_internal_energy

.. py:function:: internal_energy_from_energy(E, rho, u)
   :canonical: jf1uids.fluid_equations.fluid.internal_energy_from_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.internal_energy_from_energy

.. py:function:: pressure_from_energy(E, rho, u, gamma)
   :canonical: jf1uids.fluid_equations.fluid.pressure_from_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.pressure_from_energy

.. py:function:: total_energy_from_primitives(rho, u, p, gamma)
   :canonical: jf1uids.fluid_equations.fluid.total_energy_from_primitives

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.total_energy_from_primitives

.. py:function:: speed_of_sound(rho, p, gamma)
   :canonical: jf1uids.fluid_equations.fluid.speed_of_sound

   .. autodoc2-docstring:: jf1uids.fluid_equations.fluid.speed_of_sound
