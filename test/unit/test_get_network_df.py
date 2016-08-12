import unittest
import os
import knpackage.toolbox as knpkg
import shutil
import numpy.testing as npytest
import pandas as pd

class TestGet_network_df(unittest.TestCase):
    def setUp(self):
        print("In setUp()")

        self.network_file = "final_clean_4col.edge"
        self.config_dir = "./config_tmp"
        self.network_name = os.path.join(self.config_dir, self.network_file)
        self.network_data = "ENSG00000001631\tENSG00000000005\t3.4532771600248598e-06\tST90Q\n" + \
                            "ENSG00000092054\tENSG00000000005\t3.4532771600248598e-06\tST90Q\n"

        self.golden_output = pd.DataFrame([['ENSG00000001631','ENSG00000000005',3.4532771600248598e-06],
                                            ['ENSG00000092054','ENSG00000000005',3.4532771600248598e-06]])

    # Cleans up variables
    def tearDown(self):
        print("In tearDown()")
        del self.network_file
        del self.network_data
        del self.config_dir
        del self.golden_output

    def createFile(self, dir_name, file_name, file_content):
        os.makedirs(dir_name, mode=0o755, exist_ok=True)
        with open(os.path.join(dir_name, file_name), "w") as f:
            f.write(file_content)
        f.close()

    def test_run_file(self):
        print("In test_get_run_parameters()")
        self.createFile(self.config_dir, self.network_file, self.network_data)
        network_df = knpkg.get_network_df(self.network_name)
        npytest.assert_array_equal(network_df, self.golden_output)
        shutil.rmtree(self.config_dir)


if __name__ == '__main__':
    unittest.main()

