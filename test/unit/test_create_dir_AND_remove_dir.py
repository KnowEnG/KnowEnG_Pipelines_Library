import os
import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestCreate_dir_AND_remove_dir(TestCase):
    def setUp(self):
        self.run_parameters = {
            'test_directory': '.'}

    def tearDown(self):
        del self.run_parameters

    def test_create_dir_AND_remove_dir(self):
        """ assert that the functions work togeather to create and remove a directory
            even when files have been added
        """
        dir_name = 'tmp_test'
        dir_path = self.run_parameters['test_directory']
        new_directory_name = kn.create_dir(dir_path, dir_name)
        self.assertTrue(os.path.exists(new_directory_name), msg='create_dir function exception')
        A = np.random.rand(10, 10)
        time_stamp = '123456789'
        a_name = os.path.join(new_directory_name, 'temp_test' + time_stamp)
        with open(a_name, 'wb') as fh:
            A.dump(fh)
        A_back = np.load(a_name)
        if os.path.isfile(a_name):
            os.remove(a_name)
        A_diff = A - A_back
        A_diff = A_diff.sum()
        self.assertEqual(A_diff, 0, msg='write / read directory exception')
        kn.remove_dir(new_directory_name)
        self.assertFalse(os.path.exists(new_directory_name), msg='remove_dir function exception')

if __name__ == '__main__':
    unittest.main()