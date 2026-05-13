import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

from .error_handling_routines import ErrorHandlerIndexing as ErrorHandler
    
@pytest.mark.parametrize("Np,m", [(4, 10), (20, 6), (100, 4)])
def test_inv_occ_mat(Np, m):
    array_in = mlx_create_c.create_bos_ns(Np, m)

    expected = np.arange(array_in.shape[0])
    actual = mlx_create_c.inv_occ_mat(array_in)

    if not np.array_equal(expected, actual):
        err = ErrorHandler(Np, m, actual, expected)
        pytest.fail(err.return_error_string())
