import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

class ErrorHandler:
    def __init__(self, Np, m, Ns1, Ns, mapmat, actual, expected):
        self.Np, self.m, self.Ns1, self.Ns, self.mapmat, self.actual, self.expected = \
            Np, m, Ns1, Ns, mapmat, actual, expected
        self._find_errors()

    def _find_errors(self):
        self.errors = np.argwhere(np.any(self.actual != self.expected, axis=2))
        self.err_num = self.errors.shape[0]

    def _construct_single_error_msg(self, num: int):
        state_index = self.errors[num, 0]
        orbital_index = self.errors[num, 1]
        Ns1 = self.Ns1[state_index, :]

        target_number_state = Ns1.copy()
        target_number_state[orbital_index] += 1
        state_mapmat_points_at = self.Ns[self.mapmat[state_index, orbital_index], :]

        return (f'Error {num+1}:\n'
                f'The {self.Np-1} particle number state:\n'
                f'Ind:{state_index} Ns1:{Ns1},\n'
                f'was mapped to the wrong {self.Np} number state when attempting to\n'
                f'add a particle to the orbital with index:{orbital_index}\n'
                f'Got:      {state_mapmat_points_at}\n'
                f'Expected: {target_number_state}')

    def return_error_string(self):
        return f'{self.err_num} errors detected when testing mapmat with:\n' \
            + f'Np:{Np} and m:{m}\n\n' \
            + "\n".join((self._construct_single_error_msg(i)
                         for i in range(self.err_num)))
    
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
