import numpy as np
import numpy.testing as npt
import mlx_create_c

import pytest

class ErrorHandler:
    def __init__(self, Np, m, actual, expected):
        self.Np, self.m,  self.actual, self.expected = \
            Np, m, actual, expected
        self._find_errors()

    def _find_errors(self):
        self.errors = np.argwhere(np.any(self.actual != self.expected, axis=1))
        self.err_num = self.errors.shape[0]

    def _construct_single_error_msg(self, num: int):
        state_index = self.errors[num]

        actual_number_state = self.actual[state_index, :]
        target_number_state = self.expected[state_index, :]

        return (f'Error {num+1}:\n'
                f'Mismatch for the state {state_index}:\n'
                f'Got:      {actual_number_state}\n'
                f'Expected: {target_number_state}')

    def return_error_string(self):
        return f'{self.err_num} errors detected when testing the number states with:\n' \
            + f'Np = {self.Np} and m={self.m}:\n\n'\
            + "\n".join((self._construct_single_error_msg(i)
                         for i in range(self.err_num)))
    
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
