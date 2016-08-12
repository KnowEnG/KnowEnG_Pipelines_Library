import unittest
import knpackage.toolbox as knpkg
import os
import pandas as pd
import shutil
import numpy.testing as npytest


class TestGet_spreadsheet_df(unittest.TestCase):
    def setUp(self):
        print("In setUp()")
        self.config_dir = "./config_tmp"
        self.user_spreadsheet = "user_spreadsheet.csv"
        self.run_parameter_template = {
            'method': 'cc_cluster_nmf',
            'k': 4,
        }

    def tearDown(self):
        print("In tearDown()")
        pass

    def test_get_spreadsheet_df(self):
        print("In test_get_spreadsheet_df()")
        data = "1\t2\t3\t4\n" + \
               "5\t6\t7\t8\n" + \
               "9\t1\t1\t2\n"
        golden_output = pd.DataFrame([[1,2,3,4],[5,6,7,8],[9,1,1,2]])

        file_path = os.path.join(self.config_dir, self.user_spreadsheet)
        createFile(self.config_dir, self.user_spreadsheet, data)
        self.run_parameter_template['samples_file_name'] = file_path

        spreadsheet = knpkg.get_spreadsheet_df(self.run_parameter_template)
        npytest.assert_array_equal(golden_output,spreadsheet)
        shutil.rmtree(self.config_dir)

    def test_missing_key_in_run_parameters(self):
        print("In test_missing_key_in_run_parameters()")
        self.assertRaises(KeyError, knpkg.get_spreadsheet_df, self.run_parameter_template)

    def test_bad_user_spreadsheet(self):
        print("In test_bad_user_spreadsheet")
        data = "1\t2\t3\t4\n" + \
               "5\t6\t7\t8\n" + \
               "9\t1\t1,2\n"

        file_path = os.path.join(self.config_dir, self.user_spreadsheet)
        createFile(self.config_dir, self.user_spreadsheet, data)
        self.run_parameter_template['samples_file_name'] = file_path
        spreadsheet = knpkg.get_spreadsheet_df(self.run_parameter_template)
        print(spreadsheet)


def createFile(dir_name, file_name, file_content):
    os.makedirs(dir_name, mode=0o755, exist_ok=True)
    with open(os.path.join(dir_name, file_name), "w") as f:
        f.write(file_content)
    f.close()


if __name__ == '__main__':
    unittest.main()
