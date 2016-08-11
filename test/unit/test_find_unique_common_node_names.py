import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestNode_names(TestCase):
    def setUp(self):
        self.list_1 = [1, 2, 3]
        self.list_2 = [1, 'b', 'c']
        self.df = pd.DataFrame([1,2,3])

    def tearDown(self):
        del self.list_1[:]
        del self.list_2[:]

    def test_find_unique_node_names(self):
        ret = kn.find_unique_node_names(self.list_1, self.list_2)
        self.assertEqual(True, set(ret)==set([1, 2, 3, 'b', 'c']),
                        'incorrect union')

    def test_find_common_node_names(self):
        ret = kn.find_common_node_names(self.list_1, self.list_2)
        self.assertEqual(True, set(ret)==set([1]), 
                        'incorrect intersection')

if __name__ == '__main__':
    unittest.main()
