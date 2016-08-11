import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class TestConvert_network_df_to_sparse(TestCase):
    def test_convert_network_df_to_sparse(self):
        pg_network_df = pd.DataFrame([[1, 3, 1], [2, 4, 1]])
        res = kn.convert_network_df_to_sparse(pg_network_df, 5, 5)
        compare_result = csr_matrix(([1, 1], ([3, 4], [1, 2])), (5, 5))
        self.assertEqual(0, (res-compare_result).nnz)

if __name__ == '__main__':
    unittest.main()