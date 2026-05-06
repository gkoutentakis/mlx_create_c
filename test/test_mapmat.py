import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

from .error_handling_routines import ErrorHandlerMapMat as ErrorHandler
    
@pytest.mark.parametrize("Np,m", [(4, 10), (20, 6), (100, 4)])
def test_mapmat(Np: int, m: int):
    Ns1 = mlx_create_c.create_bos_ns(Np-1, m)
    Ns  = mlx_create_c.create_bos_ns(Np, m)
    mapmat = mlx_create_c.create_mapmat_bos(Ns1)

    actual = Ns[mapmat, :]
    expected = Ns1[:, None, :] + np.eye(m, dtype=Ns1.dtype)[None, :, :]

    if not np.array_equal(expected, actual):
        err = ErrorHandler(Np, m, Ns1, Ns, mapmat, actual, expected)
        pytest.fail(err.return_error_string())
