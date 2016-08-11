import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestSymmetrize_df(TestCase):
    def test_symmetrize_df(self):
        res = pd.DataFrame([[1, 2, 1], [3, 4, 1], [2, 1, 1], [4, 3, 1]])
        res.columns = ['node_1', 'node_2', 'wt']
        network = pd.DataFrame([[1, 2, 1], [3, 4, 1]])
        network.columns = ['node_1', 'node_2', 'wt']
        compare_result = kn.symmetrize_df(network)
        print(compare_result)
        print(res)
        self.assertEqual(True, compare_result.equals(res))

if __name__ == '__main__':
    unittest.main()