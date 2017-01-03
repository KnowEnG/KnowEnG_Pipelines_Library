import unittest
from unittest import TestCase

import knpackage.toolbox as kn

class TestCreate_timestamped_filename(TestCase):

    def test_create_timestamped_filename(self):
        """ assert that the beginning char string remains unchanged and that the
            size of the returned string is as expected
        """
        precision = None
        n_digits = 29
        name_base = 'test_string'
        name_extension = 'wie'
        tsfn = kn.create_timestamped_filename(name_base, name_extension, precision, n_digits)
        self.assertEqual(name_base, tsfn[0:11], msg='prefix name exception')
        n_chars = len(tsfn)
        self.assertEqual(name_extension, tsfn[n_chars-3:n_chars], msg='extension name exception')

        precision = 1e-15
        tsfn = kn.create_timestamped_filename(name_base, name_extension, precision, n_digits)
        self.assertEqual(name_base, tsfn[0:11], msg='prefix name exception')
        n_chars = len(tsfn)
        self.assertEqual(name_extension, tsfn[n_chars-3:n_chars], msg='extension name exception')


if __name__ == '__main__':
    unittest.main()
