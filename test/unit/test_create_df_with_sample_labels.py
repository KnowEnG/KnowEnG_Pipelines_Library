import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import numpy as np
import pandas as pd

class TestCreate_df_with_sample_labels(TestCase):
    def test_dtype_np_array(self):
        sample_names = np.array(['a', 'b', 'c'])
        labels = np.array([0, 1, 2])
        res = kn.create_df_with_sample_labels(sample_names, labels)
        compare_result = pd.DataFrame([0, 1, 2])
        compare_result.index = ['a', 'b', 'c']
        self.assertEqual(True, compare_result.equals(res))

if __name__ == '__main__':
    unittest.main()
