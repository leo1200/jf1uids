:py:mod:`jf1uids.fluid_equations.total_quantities`
==================================================

.. py:module:: jf1uids.fluid_equations.total_quantities

.. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`calculate_internal_energy <jf1uids.fluid_equations.total_quantities.calculate_internal_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_internal_energy
          :summary:
   * - :py:obj:`calculate_kinetic_energy <jf1uids.fluid_equations.total_quantities.calculate_kinetic_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_kinetic_energy
          :summary:
   * - :py:obj:`calculate_gravitational_energy <jf1uids.fluid_equations.total_quantities.calculate_gravitational_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_gravitational_energy
          :summary:
   * - :py:obj:`calculate_total_energy <jf1uids.fluid_equations.total_quantities.calculate_total_energy>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_total_energy
          :summary:
   * - :py:obj:`calculate_total_mass <jf1uids.fluid_equations.total_quantities.calculate_total_mass>`
     - .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_total_mass
          :summary:

API
~~~

.. py:function:: calculate_internal_energy(state, helper_data, gamma, config, registered_variables)
   :canonical: jf1uids.fluid_equations.total_quantities.calculate_internal_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_internal_energy

.. py:function:: calculate_kinetic_energy(state, helper_data, config, registered_variables)
   :canonical: jf1uids.fluid_equations.total_quantities.calculate_kinetic_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_kinetic_energy

.. py:function:: calculate_gravitational_energy(state, helper_data, gravitational_constant, config, registered_variables)
   :canonical: jf1uids.fluid_equations.total_quantities.calculate_gravitational_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_gravitational_energy

.. py:function:: calculate_total_energy(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, helper_data: jf1uids.data_classes.simulation_helper_data.HelperData, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], gravitational_constant: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: jf1uids.option_classes.simulation_config.SimulationConfig, registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, ]
   :canonical: jf1uids.fluid_equations.total_quantities.calculate_total_energy

   .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_total_energy

.. py:function:: calculate_total_mass(primitive_state: jf1uids.option_classes.simulation_config.STATE_TYPE, helper_data: jf1uids.data_classes.simulation_helper_data.HelperData, config: jf1uids.option_classes.simulation_config.SimulationConfig) -> jaxtyping.Float[jaxtyping.Array, ]
   :canonical: jf1uids.fluid_equations.total_quantities.calculate_total_mass

   .. autodoc2-docstring:: jf1uids.fluid_equations.total_quantities.calculate_total_mass
