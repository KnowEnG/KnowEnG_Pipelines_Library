import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

class TestMultiple_functions(TestCase):
    def test_append_column_to_spreadsheet(self):
        spreadsheet_df = pd.DataFrame([1, 0, 0, 0, 0, 0, 0], columns=['GS1'])
        len_gene = 5
        ret = kn.append_column_to_spreadsheet(spreadsheet_df, len_gene)
        comp = ret.equals(pd.DataFrame({'GS1': [1, 0, 0, 0, 0, 0, 0], 'base': [1, 1, 1, 1, 1, 0, 0]}))
        print(comp, ret)
        self.assertEqual(True, comp)

    def test_symmetrize_df(self):
        res = pd.DataFrame([[1, 2, 1], [3, 4, 1], [2, 1, 1], [4, 3, 1]])
        res.columns = ['node_1', 'node_2', 'wt']
        network = pd.DataFrame([[1, 2, 1], [3, 4, 1]])
        network.columns = ['node_1', 'node_2', 'wt']
        compare_result = kn.symmetrize_df(network)
        print(compare_result)
        print(res)
        self.assertEqual(True, compare_result.equals(res))

    def test_convert_network_df_to_sparse(self):
        pg_network_df = pd.DataFrame([[1, 3, 1], [2, 4, 1]])
        res = kn.convert_network_df_to_sparse(pg_network_df, 5, 5)
        compare_result = csr_matrix(([1, 1], ([3, 4], [1, 2])), (5, 5))
        self.assertEqual(0, (res-compare_result).nnz)

    def test_create_df_with_sample_labels(self):
        sample_names = np.array(['a', 'b', 'c'])
        labels = np.array([0, 1, 2])
        res = kn.create_df_with_sample_labels(sample_names, labels)
        compare_result = pd.DataFrame([0, 1, 2])
        compare_result.index = ['a', 'b', 'c']
        self.assertEqual(True, compare_result.equals(res))


if __name__ == '__main__':
    unittest.main()