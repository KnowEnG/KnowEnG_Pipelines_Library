import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestMap_node_names_to_index(TestCase):
    def setUp(self):
        self.network = pd.DataFrame([['a', 'b', 1], ['c', 'd', 1]],
                                    columns=['node_1', 'node_2', 'wt'])
        self.genes_map = {'a': 0, 'b': 1, 'c':2, 'd':3}

    def tearDown(self):
        del self.network
        del self.genes_map

    def test_map_node_names_to_index(self):
        ret = kn.map_node_names_to_index(self.network,
                                         self.genes_map, 'node_1')
        compare_result = pd.DataFrame([[0, 'b', 1], [2, 'd', 1]])
        compare_result.columns = ['node_1', 'node_2', 'wt']
        self.assertEqual(True, compare_result.equals(ret))

    def test_map_node_names_to_index_wrong_node(self):
        ret = kn.map_node_names_to_index(self.network,
                                         self.genes_map, 'test')

if __name__ == '__main__':
    unittest.main()