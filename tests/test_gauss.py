import numpy as np
import pytest
import sys
import os

sys.path.insert(0, "../acse_la")

from acse_la import gauss, matmul, zeromat

class TestGauss(object):
    """
    Class for testing the Gaussian elimination algorithm Gauss
    and its associated functions.
    """
    @pytest.mark.parametrize('a, b, dete, xe', [
        ([[2, 9, 4], [7, 5, 3], [6, 1, 8]],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], -360.0,
         [[-0.10277777777777776, 0.18888888888888888, -0.019444444444444438],
          [0.10555555555555554, 0.02222222222222223, -0.061111111111111116],
          [0.0638888888888889, -0.14444444444444446, 0.14722222222222223]])
    ])
    def test_gauss(self, a, b, dete, xe):
        """ Test the gauss function """
        det, x = gauss(a, b)

        assert np.isclose(det, dete)
        assert np.allclose(x, xe)

    @pytest.mark.parametrize('a, b, ab', [
        ([[1, 2, 3], [4, 5, 6]],
         [[10, 11], [20, 21], [30, 31]],
         [[140, 146], [320, 335]])
    ])
    def test_matmul(self, a, b, ab):
        """ Test the matmul function """
        a_mat_b = matmul(a, b)

        assert np.allclose(a_mat_b, ab)

    @pytest.mark.parametrize('p, q, zero', [
        (3,
         4,
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    ])
    def test_zeromat(self, p, q, zero):
        """ Test the matmul function """
        pq_zero = zeromat(p, q)

        assert np.allclose(pq_zero, zero)
