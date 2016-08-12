import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestUpdate_spreadsheet_df(TestCase):
    def setUp(self):
        self.spreadsheet = pd.DataFrame([1,2,3], index=['a', 'b', 'c'])
        self.spreadsheet_result = pd.DataFrame([1], index=['a'])
        self.gene_list = ['a']

    def tearDown(self):
        del self.spreadsheet
        del self.spreadsheet_result
        del self.gene_list

    def test_update_spreadsheet_df(self):
        ret = kn.update_spreadsheet_df(self.spreadsheet, self.gene_list)
        self.assertEqual(True, ret.equals(self.spreadsheet_result), 'test_update_spreadsheet_df failed')

if __name__ == '__main__':
    unittest.main()
