import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestAppend_column_to_spreadsheet(TestCase):
    def setUp(self):
        self.spreadsheet = pd.DataFrame({'col0': [4, 5, 6]})
        self.spreadsheet_result = pd.DataFrame({'col0': [4, 5, 6], 'col1': [1, 2, 3]})
        self.column = [1, 2, 3]
        self.col_name = 'col1'

    def tearDown(self):
        del self.spreadsheet
        del self.spreadsheet_result
        del self.column
        del self.col_name

    def test_append_column_to_spreadsheet(self):
        ret = kn.append_column_to_spreadsheet(
              self.spreadsheet, self.column, self.col_name)
        self.assertEqual(True, ret.equals(self.spreadsheet_result),
                         'test_append_column_to_spreadsheet failed')


if __name__ == '__main__':
    unittest.main()
