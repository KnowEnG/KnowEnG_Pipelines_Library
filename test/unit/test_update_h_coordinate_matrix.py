import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestUpdate_h_coordinate_matrix(TestCase):

    def test_update_h_coordinate_matrix(self):
        k = 3
        rows = 25
        cols = 6
        W = np.random.rand(rows, k)
        H = np.random.rand(k, cols)
        X = W.dot(H)
        hwx = kn.update_h_coordinate_matrix(W, X)
        dh = (np.abs(H - hwx)).sum()
        self.assertAlmostEqual(dh, 0, msg='h matrix mangled exception')

if __name__ == '__main__':
    unittest.main()