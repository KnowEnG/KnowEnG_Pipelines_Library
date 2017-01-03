import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestAppend_run_parameters_dict(TestCase):
    def setUp(self):
        self.run_parameters = {
            'rwr_max_iterations': 100,
            'rwr_convergence_tolerence': 1e-15,
            'rwr_restart_probability': 1}

    def tearDown(self):
        del self.run_parameters

    def test_append_run_parameters_dict(self):
        """ assert that key value pairs are inserted and are retrevable from the run
            parameters dictionary
        """
        run_parameters = self.run_parameters
        run_parameters = kn.append_run_parameters_dict(run_parameters, 'pi_test', np.pi)
        run_parameters = kn.append_run_parameters_dict(run_parameters, 'tea_test', 'tea')

        self.assertEqual(run_parameters['pi_test'], np.pi, msg='float value exception')
        self.assertEqual(run_parameters['tea_test'], 'tea', msg='string value exception')

if __name__ == '__main__':
    unittest.main()