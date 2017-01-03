import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestUpdate_indicator_matrix(TestCase):

    def test_update_indicator_matrix(self):
        """ assert that the indicator matrix is not loosing any digits
            Note: correctness test considered as part of linkage matrix test
        """
        n_repeats = 10
        n_test_perm = 11
        n_test_rows = 77
        A = np.zeros((n_test_rows, n_test_rows))
        running_sum = 0
        for r in range(0, n_repeats):
            running_sum += n_test_perm ** 2
            f_perm = np.random.permutation(n_test_rows)
            f_perm = f_perm[0:n_test_perm]
            A = kn.update_indicator_matrix(f_perm, A)

        self.assertEqual(A.sum(), running_sum, msg='sum of elements exception')

if __name__ == '__main__':
    unittest.main()