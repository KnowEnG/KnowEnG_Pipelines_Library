import unittest
import numpy as np
import scipy.sparse as spar
import scipy.stats as stats
from unittest import TestCase

# file version to test
import knpackage.toolbox as kn

class TestNormalize_sparse_mat_by_diagonal(TestCase):

    def test_normalize_sparse_mat_by_diagonal(self):
        """ assert that a test matrix will be "normalized" s.t. the sum of the rows
            or columns will nearly equal one
        """
        A = np.random.rand(500, 500)
        B = kn.normalize_sparse_mat_by_diagonal(spar.csr_matrix(A))
        B = B.todense()
        B2 = B ** 2
        B2 = np.sqrt(B2.sum())
        geo_mean = float(stats.gmean(B.sum(axis=1)))
        self.assertAlmostEqual(geo_mean, 1, delta=0.1)
        geo_mean = float(stats.gmean(B.T.sum(axis=1)))
        self.assertAlmostEqual(geo_mean, 1, delta=0.1)

if __name__ == '__main__':
    unittest.main()