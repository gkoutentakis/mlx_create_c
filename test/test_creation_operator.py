import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

np.seterr(divide='raise', invalid='raise', over='raise')
from .error_handling_routines import ErrorHandlerMapMat as ErrorHandler

@pytest.mark.parametrize("Np,m", [(4, 10), (20, 6), (100, 4)])
def test_creation_operator(Np: int, m: int):
    Ns1 = mlx_create_c.create_bos_ns(Np-1, m)
    Ns  = mlx_create_c.create_bos_ns(Np, m)
    mapmat_expected = mlx_create_c.create_mapmat_bos(Ns1)
    weight_matrix_expected = np.sqrt(Ns[mapmat_expected[:, :], np.arange(m)[None,:]])

    mapmat_actual, weight_actual = \
        mlx_create_c.creation_operator(np.arange(Ns1.shape[0])[:, None],
                                       np.arange(Ns1.shape[1])[None, :],
                                       Np-1, m)

    if not np.array_equal(mapmat_expected, mapmat_actual):
        err = ErrorHandler(Np, m, Ns1, Ns, mapmat_actual, mapmat_actual, mapmat_matrix_expected)
        pytest.fail(err.return_error_string())

    if not np.array_equal(weight_matrix_expected, weight_actual):
        err = ErrorHandler(Np, m, Ns1, Ns, mapmat_actual, weight_actual, weight_matrix_expected)
        pytest.fail(err.return_error_string())
