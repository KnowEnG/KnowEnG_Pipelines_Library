import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestExtract_spreadsheet_gene_names(TestCase):
    def setUp(self):
        self.spreadsheet = pd.DataFrame([1,2,3], index=['a', 'b', 'c'])
        self.spreadsheet_empty = pd.DataFrame()

    def tearDown(self):
        del self.spreadsheet
        del self.spreadsheet_empty

    def test_extract_spreadsheet_gene_names(self):
        ret = kn.extract_spreadsheet_gene_names(self.spreadsheet)
        self.assertEqual(set(ret), set(['a', 'b', 'c']), 'wrong output')

        ret = kn.extract_spreadsheet_gene_names(self.spreadsheet_empty)
        self.assertEqual(set(ret), set([]),
                         'wrong output for empty spreadsheet')

    # def test_extract_spreadsheet_gene_names_empty(self):
if __name__ == '__main__':
    unittest.main()
