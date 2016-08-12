import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import os

class TestMultiple_functions(TestCase):
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