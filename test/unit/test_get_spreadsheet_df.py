import unittest
import knpackage.toolbox as knpkg
import os
import pandas as pd
import shutil
import numpy.testing as npytest
import xmlrunner

class TestGet_spreadsheet_df(unittest.TestCase):
    def setUp(self):
        self.config_dir = "./config_tmp"
        self.user_spreadsheet = "user_spreadsheet.csv"
        self.full_file_path = os.path.join(self.config_dir, self.user_spreadsheet)
        self.run_parameter_template = {
            'method': 'cc_cluster_nmf',
            'k': 4,
            'samples_file_name': self.full_file_path
        }

    def tearDown(self):
        shutil.rmtree(self.config_dir)
        del self.user_spreadsheet
        del self.full_file_path
        del self.run_parameter_template
        del self.config_dir


    def createFile(self, dir_name, file_name, file_content):
        os.makedirs(dir_name, mode=0o755, exist_ok=True)
        with open(os.path.join(dir_name, file_name), "w") as f:
            f.write(file_content)
        f.close()

    def test_get_spreadsheet_df(self):
        data = "\tcol1\tcol2\tcol3\n" + \
               "row1\t6\t7\t8\n" + \
               "row2\t1\t1\t2\n"
        golden_output = pd.DataFrame([[6,7,8],[1,1,2]], columns=['col1','col2', 'col3'], index=['row1', 'row2'])
        self.createFile(self.config_dir, self.user_spreadsheet, data)
        spreadsheet = knpkg.get_spreadsheet_df(self.run_parameter_template)
        npytest.assert_array_equal(golden_output,spreadsheet)


if __name__ == '__main__':
    unittest.main(testRunner = xmlrunner.XMLTestRunner(output='test_reports'),
                                                        failfast=False, buffer=False, catchbreak=False)


