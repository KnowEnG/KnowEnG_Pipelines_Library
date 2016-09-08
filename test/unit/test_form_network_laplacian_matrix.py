import numpy as np
import scipy.sparse as spar
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestForm_network_laplacian_matrix(TestCase):

    def test_form_network_laplacian_matrix(self):
        """ assert that the laplacian matrix returned sums to zero in both rows
            and columns
        """
        THRESHOLD = 0.8
        A = np.random.rand(10, 10)
        A[A < THRESHOLD] = 0
        A = A + A.T
        Ld, Lk = kn.form_network_laplacian_matrix(A)
        L = Ld - Lk
        L = L.todense()
        L0 = L.sum(axis=0)
        self.assertFalse(L0.any(), msg='Laplacian row sum not equal 0')
        L1 = L.sum(axis=1)
        self.assertFalse(L1.any(), msg='Laplacian col sum not equal 0')
        Ld, Lk = kn.form_network_laplacian_matrix(spar.csr_matrix(A))
        L = Ld - Lk
        L = L.todense()
        L0 = L.sum(axis=0)
        self.assertFalse(L0.any(), msg='Laplacian row sum not equal 0')
        L1 = L.sum(axis=1)
        self.assertFalse(L1.any(), msg='Laplacian col sum not equal 0')

if __name__ == '__main__':
    unittest.main()