:py:mod:`jf1uids.data_classes.simulation_helper_data`
=====================================================

.. py:module:: jf1uids.data_classes.simulation_helper_data

.. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HelperData <jf1uids.data_classes.simulation_helper_data.HelperData>`
     - .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_helper_data <jf1uids.data_classes.simulation_helper_data.get_helper_data>`
     - .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.get_helper_data
          :summary:

API
~~~

.. py:class:: HelperData
   :canonical: jf1uids.data_classes.simulation_helper_data.HelperData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData

   .. py:attribute:: geometric_centers
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.geometric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.geometric_centers

   .. py:attribute:: volumetric_centers
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.volumetric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.volumetric_centers

   .. py:attribute:: r
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.r
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.r

   .. py:attribute:: r_hat_alpha
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.r_hat_alpha
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.r_hat_alpha

   .. py:attribute:: cell_volumes
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.cell_volumes
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.cell_volumes

   .. py:attribute:: inner_cell_boundaries
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries

   .. py:attribute:: outer_cell_boundaries
      :canonical: jf1uids.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries

.. py:function:: get_helper_data(config: jf1uids.option_classes.simulation_config.SimulationConfig, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None, padded: bool = False) -> jf1uids.data_classes.simulation_helper_data.HelperData
   :canonical: jf1uids.data_classes.simulation_helper_data.get_helper_data

   .. autodoc2-docstring:: jf1uids.data_classes.simulation_helper_data.get_helper_data
