# API Reference

This page is organized by workflow rather than by module path. For most users,
the first two sections are enough.

## Plane-wave and exchange workflows

```{eval-rst}
.. currentmodule:: quantumhall_matrixelements

.. autofunction:: get_form_factors

.. autofunction:: get_exchange_kernels

.. autofunction:: get_exchange_kernels_compressed

.. autofunction:: get_fockmatrix_constructor

.. autofunction:: get_fockmatrix_constructor_hf

.. autofunction:: build_fockmatrix_apply
```

## Symmetric-gauge workflows

```{eval-rst}
.. currentmodule:: quantumhall_matrixelements

.. autofunction:: get_guiding_center_form_factors

.. autofunction:: get_factorized_density_form_factors

.. autofunction:: get_central_onebody_matrix_elements_compressed

.. autofunction:: materialize_central_onebody_matrix

.. autofunction:: get_haldane_pseudopotentials

.. autofunction:: get_twobody_disk_from_pseudopotentials_compressed

.. autofunction:: materialize_twobody_disk_tensor
```

## Diagnostics and opposite-field helpers

```{eval-rst}
.. currentmodule:: quantumhall_matrixelements

.. autofunction:: get_form_factors_opposite_field

.. autofunction:: get_exchange_kernels_opposite_field
```

## Advanced backend-specific entry points

```{eval-rst}
.. currentmodule:: quantumhall_matrixelements

.. autofunction:: get_exchange_kernels_laguerre

.. autofunction:: get_exchange_kernels_Ogata

.. autofunction:: get_exchange_kernels_hankel

.. autofunction:: build_exchange_fock_precompute
```
