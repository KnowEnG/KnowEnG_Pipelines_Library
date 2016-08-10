import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestFind_unique_node_names(TestCase):
    def test_find_unique_node_names(self):
        list_1 = [1,2,3,'a','t']
        list_2 = ['a',3,4,'b']
        ret = kn.find_unique_node_names(list_1, list_2)
        self.assertEqual(True, set([1,2,3,4,'a','b','t'])==set(ret))

    def test_find_common_node_names(self):
        list_1 = [1,2,3,'a','t']
        list_2 = ['a',3,4,'b']
        ret = kn.find_common_node_names(list_1, list_2)
        self.assertEqual(True, set([3,'a'])==set(ret))

    def test_extract_spreadsheet_gene_names(self):
        spreadsheet_df = pd.DataFrame([1,2,3,4], index=['a', 'b', 'c', 'd'])
        ret = kn.extract_spreadsheet_gene_names(spreadsheet_df)
        self.assertEqual(True, set(['a', 'b', 'c', 'd'])==set(ret))        

    # def test_find_dropped_node_names(self):

    def test_update_spreadsheet_df(self):
        spreadsheet_df = pd.DataFrame([1,2,3,4], index=['a', 'b', 'c', 'd'])
        gene_list = ['a', 'c']
        ret = kn.update_spreadsheet_df(spreadsheet_df, gene_list)
        res = pd.DataFrame([1,3], index=['a', 'c'])
        self.assertEqual(True, ret.equals(res))

    def test_update_network_df(self):
        network = pd.DataFrame({'a': [1,2,3,4]})
        nodes_list = [2,3]
        node_id = 'a'
        ret = kn.update_network_df(network, nodes_list, node_id)
        res = pd.DataFrame({'a': [2,3]})
        res.index=[1,2]
        self.assertEqual(True, ret.equals(res))

if __name__ == '__main__':
    unittest.main()