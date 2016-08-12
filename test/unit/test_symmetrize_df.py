import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestSymmetrize_df(TestCase):
    def setUp(self):
        self.network_none = None
        self.network = pd.DataFrame([[1, 3, 1], [2, 4, 1]],
                                    columns=['node_1', 'node_2', 'wt'])
        self.network_empty = pd.DataFrame({'node_1': [], 'node_2': [],
                                           'wt': []})
        self.network_wrong_col = pd.DataFrame([[1, 3, 1], [2, 4, 1]],
                                    columns=['a', 'b', 'c'])
    def tearDown(self):
        del self.network
        del self.network_empty
        del self.network_wrong_col

    def test_symmetrize_df(self):
        ret = kn.symmetrize_df(self.network)
        compare_ret = pd.DataFrame([[1, 3, 1], [2, 4, 1], [3, 1, 1], [4, 2, 1]],
                                    columns=['node_1', 'node_2', 'wt'])
        self.assertEqual(True, compare_ret.equals(ret))
        
    def test_symmetrize_df_empty(self):
        ret = kn.symmetrize_df(self.network_empty)
        compare_ret = pd.DataFrame({'node_1': [], 'node_2': [], 'wt': []})
        self.assertEqual(True, compare_ret.equals(ret))

    def test_symmetrize_df_wrong_column_names(self):
        ret = kn.symmetrize_df(self.network_wrong_col)
        print("wrong column names")

    def test_symmetrize_df_none(self):
        ret = kn.symmetrize_df(self.network_none)
        print("need input")

if __name__ == '__main__':
    unittest.main()