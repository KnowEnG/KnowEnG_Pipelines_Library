import unittest
import os
import knpackage.toolbox as knpkg
import shutil
import yaml

class TestGet_run_parameters(unittest.TestCase):
    def setUp(self):
        print("In setUp()")

        def createFile(dir_name, file_name, file_content):
            os.makedirs(dir_name, mode=0o755, exist_ok=True)
            with open(os.path.join(dir_name, file_name), "w") as f:
                f.write(file_content)
            f.close()

        f_bad_context = "---\n" + \
                        "method	cc_cluster_nmf\n" + \
                        "k 4\n"

        f_good_context = yaml.dump(dict(method="cc_cluster_nmf", k=4),
                                   default_flow_style=True)

        self.good_run_file = "good_run_file.yml"
        self.bad_run_file = "bad_run_file.yml"
        self.config_dir = "./config_tmp"
        self.f_bad = createFile(self.config_dir, self.bad_run_file, f_bad_context)
        self.f_good = createFile(self.config_dir, self.good_run_file, f_good_context)
        self.golden_output = {'method': 'cc_cluster_nmf',
                         'k': 4,
                         'run_directory': self.config_dir,
                         'run_file': self.good_run_file}

    def tearDown(self):
        print("In tearDown()")
        shutil.rmtree(self.config_dir)

    def test_good_run_file(self):
        print("In test_get_run_parameters()")
        run_parameters = knpkg.get_run_parameters(self.config_dir, self.good_run_file)
        self.assertDictEqual(run_parameters, self.golden_output, msg="Successfully tested good run file!")

    def test_bad_run_file(self):
        print("In test_bad_run_file()")


    def test_bad_null_input_parameters(self):
        print("In test_bad_null_input_parameters()")


if __name__ == '__main__':
    unittest.main()
