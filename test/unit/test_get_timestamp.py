import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestGet_timestamp(TestCase):

    def test_get_timestamp(self):
        """ assert that the default size of the timestamp string is 16 chars and
            that sequential calls produce differnt results
        """
        n_default_chars = 16
        stamp_time = 1e6
        tstr = kn.get_timestamp(stamp_time)
        tstr2 = kn.get_timestamp()

        self.assertEqual(len(tstr), n_default_chars, msg='string return size unexpected')
        self.assertNotEqual(tstr, tstr2, msg='successive calls same error')

if __name__ == '__main__':
    unittest.main()