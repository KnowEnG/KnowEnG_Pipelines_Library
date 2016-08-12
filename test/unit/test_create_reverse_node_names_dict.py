from unittest import TestCase
import unittest
import knpackage.toolbox as kn

class TestCreate_reverse_node_names_dict(TestCase):
    def setUp(self):
        self.dictionary = {'a': 0, 'b': 1, 'c': 2}
        self.dictionary_empty = {}

    def tearDown(self):
        del self.dictionary
        del self.dictionary_empty

    def test_create_reverse_node_names_dict(self):
        ret = kn.create_reverse_node_names_dict(self.dictionary)
        self.assertEqual(ret, {0: 'a', 1: 'b', 2: 'c'}, 'wrong output')

    def test_create_reverse_node_names_dict_empty(self):
        ret = kn.create_reverse_node_names_dict(self.dictionary_empty)
        self.assertEqual(ret, {}, 'wrong output for empty dictionary')

if __name__ == '__main__':
    unittest.main()