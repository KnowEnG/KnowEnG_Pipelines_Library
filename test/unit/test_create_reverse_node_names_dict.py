from unittest import TestCase
import unittest
import knpackage.toolbox as kn

class TestCreate_reverse_node_names_dict(TestCase):
    def test_dict_size_one(self):
        res = kn.create_reverse_node_names_dict({'a': 0})
        compare_res = {0: 'a'}
        self.assertEqual(res, compare_res)
    def test_dict_size_three_mix_type(self):
        res = kn.create_reverse_node_names_dict({'a': 0, 1: 'b', 2: 8})
        compare_res = {0: 'a', 'b': 1, 8: 2}
        self.assertEqual(res, compare_res)
    def test_dict_empty(self):
        res = kn.create_reverse_node_names_dict({})
        compare_res = {}
        self.assertEqual(res, compare_res)

if __name__ == '__main__':
    unittest.main()