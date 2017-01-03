import unittest
from unittest import TestCase
import numpy as np
import scipy.sparse as spar

import knpackage.toolbox as kn

class TestSmooth_matrix_with_rwr(TestCase):
    def setUp(self):
        self.run_parameters = {
            'rwr_max_iterations': 100,
            'rwr_convergence_tolerence': 1e-15,
            'rwr_restart_probability': 1}

    def tearDown(self):
        del self.run_parameters

    def test_smooth_matrix_with_rwr(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 32
        F0 = np.eye(2)
        A = spar.csr_matrix(((np.eye(2) + np.ones(2)) / 3))

        F_exact = np.ones((2, 2)) * 0.5
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, self.run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS)

        T = (np.abs(F_exact - F_calculated))

        self.assertAlmostEqual(T.sum(), 0)

    def test_smooth_matrix_with_rwr_non_sparse(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 32
        F0 = np.eye(2)
        A = (np.eye(2) + np.ones((2, 2))) / 3
        F_exact = np.ones((2, 2)) * 0.5
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, self.run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS)

        T = (np.abs(F_exact - F_calculated))

        self.assertAlmostEqual(T.sum(), 0)

    def test_smooth_matrix_with_rwr_single_vector(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 31
        F0 = np.array([1.0, 0.0])
        A = (np.eye(2) + np.ones((2, 2))) / 3

        F_exact = np.array([0.5, 0.5])
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, self.run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS, msg='minor difference')
        T = (np.abs(F_exact - F_calculated))
        self.assertAlmostEqual(T.sum(), 0)

if __name__ == '__main__':
    unittest.main()
