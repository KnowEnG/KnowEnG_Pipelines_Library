import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestUpdate_linkage_matrix(TestCase):

    def test_update_linkage_matrix(self):
        """ create a consensus matrix by sampling a synthesized set of clusters
            assert that the clustering is equivalent
        """
        n_samples = 11
        n_clusters = 3
        cluster_set = np.int_(np.ones(n_samples))
        for r in range(0, n_samples):
            cluster_set[r] = int(np.random.randint(n_clusters))

        n_repeats = 100
        n_test_perm = 5
        n_test_rows = n_samples
        I = np.zeros((n_test_rows, n_test_rows))
        M = np.zeros((n_test_rows, n_test_rows))

        for r in range(0, n_repeats):
            f_perm = np.random.permutation(n_test_rows)
            f_perm = f_perm[0:n_test_perm]
            cluster_p = cluster_set[f_perm]
            I = kn.update_indicator_matrix(f_perm, I)
            M = kn.update_linkage_matrix(cluster_p, f_perm, M)

        CC = M / np.maximum(I, 1e-15)

        for s in range(0, n_clusters):
            s_dex = cluster_set == s
            c_c = CC[s_dex, :]
            c_c = c_c[:, s_dex]
            n_check = c_c - 1
            self.assertEqual(n_check.sum(), 0, msg='cluster grouping exception')

if __name__ == '__main__':
    unittest.main()