from unittest import TestCase
import unittest
import knpackage.toolbox as kn


class TestCreate_reverse_node_names_dict(TestCase):
    def test_create_reverse_node_names_dict(self):
        res = kn.create_reverse_node_names_dict({'a': 0, 'b': 1})
        compare_res = {0: 'a', 1: 'b'}
        self.assertEqual(res, compare_res)

if __name__ == '__main__':
    unittest.main()