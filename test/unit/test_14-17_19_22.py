import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import os

class TestMultiple_functions(TestCase):
    # def test_append_column_to_spreadsheet(self):
    #     spreadsheet_df = pd.DataFrame([1, 0, 0, 0, 0, 0, 0], columns=['GS1'])
    #     len_gene = 5
    #     ret = kn.append_column_to_spreadsheet(spreadsheet_df, len_gene)
    #     comp = ret.equals(pd.DataFrame({'GS1': [1, 0, 0, 0, 0, 0, 0], 'base': [1, 1, 1, 1, 1, 0, 0]}))
    #     print(comp, ret)
    #     self.assertEqual(True, comp)
    #
    #
    # def test_convert_network_df_to_sparse(self):
    #     pg_network_df = pd.DataFrame([[1, 3, 1], [2, 4, 1]])
    #     res = kn.convert_network_df_to_sparse(pg_network_df, 5, 5)
    #     compare_result = csr_matrix(([1, 1], ([3, 4], [1, 2])), (5, 5))
    #     self.assertEqual(0, (res-compare_result).nnz)
    #
    # def test_create_df_with_sample_labels(self):
    #     sample_names = np.array(['a', 'b', 'c'])
    #     labels = np.array([0, 1, 2])
    #     res = kn.create_df_with_sample_labels(sample_names, labels)
    #     compare_result = pd.DataFrame([0, 1, 2])
    #     compare_result.index = ['a', 'b', 'c']
    #     self.assertEqual(True, compare_result.equals(res))
    #
    # def test_map_node_names_to_index_node_1(self):
    #     network = pd.DataFrame([['a', 'b', 1], ['c', 'd', 1]])
    #     network.columns = ['node_1', 'node_2', 'wt']
    #     genes_map = {'a': 0, 'b': 1, 'c':2, 'd':3}
    #     res = kn.map_node_names_to_index(network, genes_map, 'node_1')
    #     compare_result = pd.DataFrame([[0, 'b', 1], [2, 'd', 1]])
    #     compare_result.columns = ['node_1', 'node_2', 'wt']
    #     self.assertEqual(True, compare_result.equals(res))
    #
    # def test_map_node_names_to_index_node_2(self):
    #     network = pd.DataFrame([['a', 'b', 1], ['c', 'd', 1]])
    #     network.columns = ['node_1', 'node_2', 'wt']
    #     genes_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    #     res = kn.map_node_names_to_index(network, genes_map, 'node_2')
    #     compare_result = pd.DataFrame([['a', 1, 1], ['c', 3, 1]])
    #     compare_result.columns = ['node_1', 'node_2', 'wt']
    #     self.assertEqual(True, compare_result.equals(res))

    def test_save_df(self):
        input_df = pd.DataFrame([[1, 2], [3, 4]])
        kn.save_df(input_df, os.getcwd(), 'test_file')
        res = pd.read_csv('test_file', sep='\t', header=None, index=None)
        print(res)
        print(input_df)
        self.assertEqual(True, input_df.equals(res))


    def test_smooth_matrix_with_rwr(self):
        restart = np.transpose(np.array([[1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]]))
        network_sparse = csr_matrix(
            ([0.14999999999999999, 0.16250000000000001, 0.14999999999999999, 0.1875,
              0.16250000000000001, 0.1875, 0.083333333333333301, 0.083333333333333301,
              0.083333333333333301, 0.083333333333333301, 0.083333333333333301,
              0.083333333333333301, 0.083333333333333301, 0.083333333333333301,
              0.083333333333333301, 0.083333333333333301, 0.083333333333333301,
              0.083333333333333301],
             ([1, 2, 0, 2, 0, 1, 6, 5, 5, 6, 5, 6, 1, 2, 3, 0, 2, 4],
              [0, 0, 1, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6])), shape=(7, 7))
        run_parameters = {'number_of_iteriations_in_rwr': 500, 'it_max': 10000,
                          'restart_tolerance': 0.0001, 'restart_probability': 0.5}
        ret = kn.smooth_matrix_with_rwr(normalize(restart, norm='l1', axis=0),
                                        normalize(network_sparse, norm='l1', axis=0), run_parameters)[0]
        ret = np.around(ret, decimals=3)
        res = np.transpose(np.array([[0.5646, 0.142, 0.1658, 0.005, 0.0132, 0.0299, 0.0794],
                                     [0.1824, 0.1883, 0.2106, 0.1156, 0.1157, 0.0934, 0.094]]))
        comp = (ret == np.round(res, decimals=3)).all()
        self.assertEqual(True, comp)

if __name__ == '__main__':
    unittest.main()