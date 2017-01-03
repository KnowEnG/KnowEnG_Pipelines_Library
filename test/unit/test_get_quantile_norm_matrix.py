import unittest
import numpy as np
from unittest import TestCase

import knpackage.toolbox as kn

class TestGet_quantile_norm_matrix(TestCase):
    def test_get_quantile_norm_matrix(self):
        a = np.array([[7.0, 5.0], [3.0, 1.0], [1.0, 7.0]])
        aQN = np.array([[7.0, 4.0], [4.0, 1.0], [1.0, 7.0]])
        qn1 = kn.get_quantile_norm_matrix(a)

        self.assertEqual(sum(sum(qn1 != aQN)), 0, 'Quantile Norm 1 Not Equal')

if __name__ == '__main__':
    unittest.main()