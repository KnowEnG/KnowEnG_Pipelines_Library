import unittest
from unittest import TestCase
import knpackage.toolbox as kn


class TestCreate_node_names_dict(TestCase):
    def test_string_keys(self):
        node_names = ['a', 'b', 'c']
        res = kn.create_node_names_dict(node_names, start_value=0)
        compare_res = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(res, compare_res)
    def test_int_keys(self):
        node_names = [4, 5, 6]
        res = kn.create_node_names_dict(node_names, start_value=0)
        compare_res = {4: 0, 5: 1, 6: 2}
        self.assertEqual(res, compare_res)
    def test_mix_keys_start_nonzero(self):
        node_names = [4, "test1", 6, "test2"]
        res = kn.create_node_names_dict(node_names, start_value=8)
        compare_res = {4: 8, "test1": 9, 6: 10, "test2":11}
        self.assertEqual(res, compare_res)

    def test_empty_dict(self):
        node_names = []
        res = kn.create_node_names_dict(node_names, start_value=0)
        compare_res = {}
        self.assertEqual(res, compare_res)

if __name__ == '__main__':
    unittest.main()