import unittest
from unittest import TestCase
import knpackage.toolbox as kn


class TestCreate_node_names_dict(TestCase):
    def setUp(self):
        self.node_names = ['a', 'b', 'c']
        self.node_names_empty = []
        self.node_names_start_positive = ['a', 8, 'b']
        self.node_names_start_negative = ['a', 8, 'b']

    def tearDown(self):
        del self.node_names
        del self.node_names_empty
        del self.node_names_start_positive
        del self.node_names_start_negative

    def test_create_node_names_dict(self):
        ret = kn.create_node_names_dict(self.node_names, start_value=0)
        self.assertEqual(ret, {'a': 0, 'b': 1, 'c': 2}, 'wrong output')

    def test_create_node_names_dict_empty(self):
        ret = kn.create_node_names_dict(self.node_names_empty, start_value=0)
        self.assertEqual(ret, {}, 'wrong output for empty node_names')

    def test_create_node_names_dict_start_positive(self):
        ret = kn.create_node_names_dict(self.node_names_start_positive, start_value=5)
        self.assertEqual(ret, {'a': 5, 8: 6, 'b':7},
                         'wrong output for node_names with positive start value')

    def test_create_node_names_dict_start_negative(self):
        ret = kn.create_node_names_dict(self.node_names_start_negative, start_value=-3)
        self.assertEqual(ret, {'a': -3, 8: -2, 'b': -1},
                         'wrong output for node_names with negative start value')

if __name__ == '__main__':
    unittest.main()