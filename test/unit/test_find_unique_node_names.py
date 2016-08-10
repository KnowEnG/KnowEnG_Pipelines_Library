from unittest import TestCase
import knpackage as kn

class TestFind_unique_node_names(TestCase):
    def test_find_unique_node_names(self):
        list_1 = [1,2,3,a,t]
        list_2 = [a,3,4,b]
        ret = find_unique_node_names(list_1, list_2)
        self.assertEqual(True, set([3,a])==set(ret))
    