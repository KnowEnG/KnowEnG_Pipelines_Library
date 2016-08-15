import unittest
import pandas as pd
import knpackage.toolbox as knpkg

class TestExtract_network_node_names(unittest.TestCase):
    def setUp(self):
        self.network_df_two_col = pd.DataFrame([['ENSG00000001631','ENSG00000000005'],
                                        ['ENSG00000092054','ENSG00000000006']])
        self.network_df_three_col = pd.DataFrame([['ENSG00000001631', 'ENSG00000000005', 3.4532771600248598e-06],
                                             ['ENSG00000092054', 'ENSG00000000006', 3.4532771600248598e-06]])
        self.network_df_emtpy = pd.DataFrame()
        self.golden_output_col0 = ['ENSG00000001631', 'ENSG00000092054']
        self.golden_output_col1 = ['ENSG00000000005', 'ENSG00000000006']

    def tearDown(self):
        del self.network_df_three_col
        del self.network_df_two_col
        del self.network_df_emtpy
        del self.golden_output_col0
        del self.golden_output_col1

    def test_extract_network_node_names_good(self):
        nodelist_a, nodelist_b = knpkg.extract_network_node_names(self.network_df_three_col)
        self.assertListEqual(sorted(nodelist_a), sorted(self.golden_output_col0))
        self.assertListEqual(sorted(nodelist_b), sorted(self.golden_output_col1))

    # def test_extract_network_node_names_bad(self):
    #     ret_value = knpkg.extract_network_node_names(self.network_df_two_col)
    #     self.assertEqual(ret_value, False)
    #
    # def test_extract_network_node_none_input(self):
    #     ret_value = knpkg.extract_network_node_names(self.network_df_emtpy)
    #     self.assertEqual(ret_value, False)


if __name__ == '__main__':
    unittest.main()



