import unittest
from unittest import TestCase
import knpackage.toolbox as kn
import pandas as pd

class TestUpdate_network_df(TestCase):
    def setUp(self):
        self.network = pd.DataFrame(
                       [1,2,3], index=['a', 'b', 'c'], columns=['first'])
        self.nodes_list = [1,2]
        self.node_id = 'first'
        self.network_result = pd.DataFrame([1, 2], index=['a', 'b'], columns=['first'])

    def tearDown(self):
        del self.network
        del self.nodes_list
        del self.node_id
        del self.network_result

    def test_update_network_df(self):
        ret = kn.update_network_df(
              self.network, self.nodes_list, self.node_id)
        self.assertEqual(True, ret.equals(self.network_result),
                         'test_update_network_df failed')

if __name__ == '__main__':
    unittest.main()
