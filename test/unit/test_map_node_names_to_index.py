import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd


class TestMap_node_names_to_index(TestCase):
    def test_node_1(self):
        network = pd.DataFrame([['a', 'b', 1], ['c', 'd', 1]])
        network.columns = ['node_1', 'node_2', 'wt']
        genes_map = {'a': 0, 'b': 1, 'c':2, 'd':3}
        res = kn.map_node_names_to_index(network, genes_map, 'node_1')
        compare_result = pd.DataFrame([[0, 'b', 1], [2, 'd', 1]])
        compare_result.columns = ['node_1', 'node_2', 'wt']
        self.assertEqual(True, compare_result.equals(res))
    def test_node_2(self):
        network = pd.DataFrame([['a', 'b', 1], ['c', 'd', 1]])
        network.columns = ['node_1', 'node_2', 'wt']
        genes_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        res = kn.map_node_names_to_index(network, genes_map, 'node_2')
        compare_result = pd.DataFrame([['a', 1, 1], ['c', 3, 1]])
        compare_result.columns = ['node_1', 'node_2', 'wt']
        self.assertEqual(True, compare_result.equals(res))


if __name__ == '__main__':
    unittest.main()