import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd
from scipy.sparse import csr_matrix


class TestConvert_network_df_to_sparse(TestCase):
    def setUp(self):
        self.pg_network_df = pd.DataFrame([[1, 3, 1], [2, 4, 1]])
        self.pg_network_df_empty = pd.DataFrame({0: [], 1: [], 2: []})

    def tearDown(self):
        del self.pg_network_df
        del self.pg_network_df_empty

    def test_convert_network_df_to_sparse(self):
        ret = kn.convert_network_df_to_sparse(self.pg_network_df, 5, 5)
        compare_result = csr_matrix(([1, 1], ([3, 4], [1, 2])), (5, 5))
        self.assertEqual(0, (ret-compare_result).nnz, "wrong output")

    # def test_convert_empty_network_df_to_sparse(self):
    #     ret = kn.convert_network_df_to_sparse(self.pg_network_df_empty, 0, 0)
    #
    # def test_sparse_size(self):
    #     ret = kn.convert_network_df_to_sparse(self.pg_network_df, 5.5, 5)
    #     ret = kn.convert_network_df_to_sparse(self.pg_network_df, 4, 4)

if __name__ == '__main__':
    unittest.main()
