:py:mod:`jf1uids._physics_modules._cosmic_rays.cr_fluid_equations`
==================================================================

.. py:module:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations

.. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`total_energy_from_primitives_with_crs <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_energy_from_primitives_with_crs>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_energy_from_primitives_with_crs
          :summary:
   * - :py:obj:`gas_pressure_from_primitives_with_crs <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gas_pressure_from_primitives_with_crs>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gas_pressure_from_primitives_with_crs
          :summary:
   * - :py:obj:`total_pressure_from_conserved_with_crs <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_pressure_from_conserved_with_crs>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_pressure_from_conserved_with_crs
          :summary:
   * - :py:obj:`speed_of_sound_crs <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.speed_of_sound_crs>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.speed_of_sound_crs
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`gamma_gas <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_gas>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_gas
          :summary:
   * - :py:obj:`gamma_cr <jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_cr>`
     - .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_cr
          :summary:

API
~~~

.. py:data:: gamma_gas
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_gas
   :value: None

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_gas

.. py:data:: gamma_cr
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_cr
   :value: None

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gamma_cr

.. py:function:: total_energy_from_primitives_with_crs(primitive_state: jaxtyping.Float[jaxtyping.Array, num_vars num_cells], registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, num_cells]
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_energy_from_primitives_with_crs

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_energy_from_primitives_with_crs

.. py:function:: gas_pressure_from_primitives_with_crs(primitive_state: jaxtyping.Float[jaxtyping.Array, num_vars num_cells], registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, num_cells]
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gas_pressure_from_primitives_with_crs

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.gas_pressure_from_primitives_with_crs

.. py:function:: total_pressure_from_conserved_with_crs(conserved_state: jaxtyping.Float[jaxtyping.Array, num_vars num_cells], registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, num_cells]
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_pressure_from_conserved_with_crs

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.total_pressure_from_conserved_with_crs

.. py:function:: speed_of_sound_crs(primitive_state: jaxtyping.Float[jaxtyping.Array, num_vars num_cells], registered_variables: jf1uids.fluid_equations.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, num_cells]
   :canonical: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.speed_of_sound_crs

   .. autodoc2-docstring:: jf1uids._physics_modules._cosmic_rays.cr_fluid_equations.speed_of_sound_crs
