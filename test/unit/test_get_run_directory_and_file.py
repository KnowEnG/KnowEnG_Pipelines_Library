import unittest
import knpackage.toolbox as knpkg
import xmlrunner
class TestGet_run_directory_and_file(unittest.TestCase):
    def setUp(self):
        print("In setUp()")

    def tearDown(self):
        print("In tearDown()")


if __name__ == '__main__':
    unittest.main(testRunner = xmlrunner.XMLTestRunner(output='test_reports'),
                                                        failfast=False, buffer=False, catchbreak=False)