import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd
import os

class TestSave_df(TestCase):
    def setUp(self):
        self.result_df = pd.DataFrame({'a': [1, 2, 3]})
        self.tmp_dir = '.'
        self.file_name = 'test.txt'
        self.test_df = pd.DataFrame({'a': [1, 2, 3]})

    def tearDown(self):
        del self.result_df
        del self.tmp_dir
        del self.file_name
        del self.test_df

    def test_save_df(self):
        kn.save_df(self.result_df, self.tmp_dir, self.file_name)
        ret = pd.read_csv(os.path.join(self.tmp_dir, self.file_name),
              sep='\t', header=0, index_col=False)
        os.remove(os.path.join(self.tmp_dir, self.file_name))
        self.assertEqual(True, ret.equals(self.test_df),
                         'test_save_df failed')

if __name__ == '__main__':
    unittest.main()
