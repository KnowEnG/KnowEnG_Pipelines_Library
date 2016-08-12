import unittest
import os
import shutil
import yaml
import knpackage.toolbox as knpkg


class TestGet_run_parameters(unittest.TestCase):
    def setUp(self):
        print("In setUp()")

        self.f_context = yaml.dump(dict(method="cc_cluster_nmf", k=4),
                                   default_flow_style=True)

        self.run_file = "run_file.yml"
        self.config_dir = "./config_tmp"
        self.golden_output = {
            'method': 'cc_cluster_nmf',
            'k': 4,
            'run_directory': self.config_dir,
            'run_file': self.run_file
        }

    def tearDown(self):
        print("In tearDown()")
        del self.f_context
        del self.run_file
        del self.config_dir
        del self.golden_output
        shutil.rmtree(self.config_dir)

    def createFile(self, dir_name, file_name, file_content):
        os.makedirs(dir_name, mode=0o755, exist_ok=True)
        with open(os.path.join(dir_name, file_name), "w") as f:
            f.write(file_content)
        f.close()

    def test_run_file(self):
        print("In test_get_run_parameters()")
        self.createFile(self.config_dir, self.run_file, self.f_context)
        run_parameters = knpkg.get_run_parameters(self.config_dir, self.run_file)
        self.assertDictEqual(run_parameters, self.golden_output)



if __name__ == '__main__':
    unittest.main()


