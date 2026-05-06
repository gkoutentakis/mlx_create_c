import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

from .error_handling_routines import ErrorHandlerIndexing as ErrorHandler
    
# @pytest.mark.parametrize("Np,m", [(5, 3)])
@pytest.mark.parametrize("Np,m", [(4, 10), (20, 6), (100, 4)])
def test_mapmat_explicit_loop(Np: int, m: int):
    expected = mlx_create_c.create_bos_ns(Np, m)

    actual = np.zeros_like(expected)
    for i in range(expected.shape[0]):
        actual[i, :] = mlx_create_c.configuration_from_state_index(i, Np, m)
        
    if not np.array_equal(expected, actual):
        err = ErrorHandler(Np, m, actual, expected)
        pytest.fail(err.return_error_string())

@pytest.mark.parametrize("Np,m", [(4, 10), (20, 6), (100, 4)])
def test_mapmat_vector(Np: int, m: int):
    expected = mlx_create_c.create_bos_ns(Np, m)

    indices = np.arange(expected.shape[0])
    actual = mlx_create_c.configuration_from_state_index(indices, Np, m)
        
    if not np.array_equal(expected, actual):
        err = ErrorHandler(Np, m, actual, expected)
        pytest.fail(err.return_error_string())
