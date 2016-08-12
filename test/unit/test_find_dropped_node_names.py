import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestFind_dropped_node_names(TestCase):
    def setUp(self):
        self.spreadsheet_df = pd.DataFrame([1,2,3], index=['a', 'b', 'c'])
        self.unique_gene_names = ['a', 'b']
        self.droplist = ['c']

    def tearDown(self):
        del self.spreadsheet_df
        del self.unique_gene_names
        del self.droplist

    def test_find_dropped_node_names(self):
        ret = kn.find_dropped_node_names(self.spreadsheet_df, self.unique_gene_names)
        self.assertEqual(ret.index.values, self.droplist, 'test_find_dropped_node_names failed')


if __name__ == '__main__':
    unittest.main()