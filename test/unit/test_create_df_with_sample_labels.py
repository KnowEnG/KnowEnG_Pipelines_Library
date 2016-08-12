import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import numpy as np
import pandas as pd

class TestCreate_df_with_sample_labels(TestCase):
    def setUp(self):
        self.sample_names = np.array(['a', 'b', 'c'])
        self.labels = np.array([0, 1, 2])
        self.sample_names_empty = []
        self.labels_empty = []

    def tearDown(self):
        del self.sample_names
        del self.labels

    def test_create_df_with_sample_labels(self):
        ret = kn.create_df_with_sample_labels(self.sample_names, self.labels)
        compare_result = pd.DataFrame([0, 1, 2])
        compare_result.index = ['a', 'b', 'c']
        self.assertEqual(True, compare_result.equals(ret))

    def test_create_empty_df_with_sample_labels(self):
        ret = kn.create_df_with_sample_labels(self.sample_names_empty,
                                              self.labels_empty)
        compare_result = pd.DataFrame()
        self.assertEqual(True, compare_result.equals(ret))


if __name__ == '__main__':
    unittest.main()