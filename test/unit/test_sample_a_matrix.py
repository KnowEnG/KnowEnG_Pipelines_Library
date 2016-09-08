import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestSample_a_matrix(TestCase):

    def test_sample_a_matrix(self):
        """ assert that the random sample is of the propper size, the
            permutation points to the correct columns and that the number of
            rows set to zero is correct.
        """
        n_test_rows = 11
        n_test_cols = 5
        pct_smpl = 0.6
        n_zero_rows = int(np.round(n_test_rows * (1 - pct_smpl)))
        n_smpl_cols = int(np.round(n_test_cols * pct_smpl))
        epsilon_sum = max(n_test_rows, n_test_cols) * 1e-15
        A = np.random.rand(n_test_rows, n_test_cols) + epsilon_sum
        B, P = kn.sample_a_matrix(A, pct_smpl, pct_smpl)
        self.assertEqual(B.shape[1], P.size, msg='permutation size not equal columns')
        self.assertEqual(P.size, n_smpl_cols, msg='number of sample columns exception')
        perm_err_sum = 0
        n_zero_err_sum = 0
        B_col = 0
        for A_col in P:
            n_zeros = (np.int_(B[:, B_col] == 0)).sum()
            if n_zeros != n_zero_rows:
                n_zero_err_sum += 1
            C = A[:, A_col] - B[:, B_col]
            C[B[:, B_col] == 0] = 0
            B_col += 1
            if C.sum() > epsilon_sum:
                perm_err_sum += 1

        self.assertEqual(n_zero_err_sum, 0, msg='number of zero columns exception')
        self.assertEqual(perm_err_sum, 0, msg='permutation index exception')

if __name__ == '__main__':
    unittest.main()