# -*- coding: utf-8 -*-
"""     lanier4@illinois.edu    """

import unittest
import os
import numpy as np
import scipy.sparse as spar
import scipy.stats as stats

# file version to test
import knpackage.toolbox as kn


def synthesize_random_network(network_dim, n_nodes):
    """ symmetric random adjacency matrix from random set of nodes
    Args:
        network_dim: number of rows and columns in the symmetric output matrix
        n_nodes: number of connections (approximate because duplicates are ignored)
    Returns:
        network: a symmetric adjacency matrix (0 or 1 in network_dim x network_dim matrix)
    """
    network = np.zeros((network_dim, network_dim))
    col_0 = np.random.randint(0, network_dim, n_nodes)
    col_1 = np.random.randint(0, network_dim, n_nodes)
    for node in range(0, n_nodes):
        if col_0[node] != col_1[node]:
            network[col_0[node], col_1[node]] = 1
    network = network + network.T
    network[network != 0] = 1

    return network


def get_cluster_indices_list(a_arr):
    """ get the list of sets of positive integers in the input array where a set
        is the index of where equal values occur for all equal values in the array

    Args:
        a_arr: array of positive integers

    Returns:
        cluster_list: list of lists where each list is the indecies of the members
            of that set, and the lists are ordered by the first member of each.
    """
    idx_arr = np.arange(0, a_arr.size)
    a_arr_unique = np.unique(a_arr)
    tmp_list = []
    for v in a_arr_unique:
        tmp_list.append(idx_arr[a_arr == v])

    len_tmp_list = len(tmp_list)
    first_member_array = np.int_(np.zeros(len_tmp_list))
    for m in range(0, len_tmp_list):
        tmp = tmp_list[m]
        first_member_array[m] = int(tmp[0])

    list_order = np.int_(np.argsort(first_member_array))
    cluster_list = []
    for t in list_order:
        cluster_list.append(tmp_list[t])

    return cluster_list


def sets_a_eq_b(a, b):
    """ check that all indices of equal values in a
        are same sets as indices of equal values in b
    Args:
        a: array of cluster assignments
        b: array of cluster assignments - same size or will return false
    Returns:
        True or False: array a indices of equal value
            are the same as array b indices of equal values
    """
    a_u = np.unique(a)
    b_u = np.unique(b)
    if len(a_u) != len(b_u):
        return False
    else:
        a_list = get_cluster_indices_list(a)
        b_list = get_cluster_indices_list(b)
        if len(b) != len(a):
            return False
        else:
            n_here = 0
            for a_set in a_list:
                if (len(a_set) != len(b_list[n_here])):
                    return False
                elif sum(np.int_(a_set != b_list[n_here])) != 0:
                    return False
                else:
                    n_here += 1
    return True


class toolbox_test(unittest.TestCase):
    def get_run_parameters(self):
        run_parameters = {'test_directory': '/Users/del/AllCodeBigData/KnowEnG_tbx_test',
                          'k': 3,
                          'number_of_iteriations_in_rwr': 100,
                          'obj_fcn_chk_freq': 50,
                          'it_max': 10000,
                          'h_clust_eq_limit': 100,
                          'restart_tolerance': 0.0001,
                          'lmbda': 1400,
                          'percent_sample': 0.8,
                          'number_of_bootstraps': 3,
                          'display_clusters': 1,
                          'restart_probability': 0.7,
                          'verbose': 1,
                          'use_now_name': 1000000}

        return run_parameters

    def test_get_quantile_norm_matrix(self):
        a = np.array([[7.0, 5.0], [3.0, 1.0], [1.0, 7.0]])
        aQN = np.array([[7.0, 4.0], [4.0, 1.0], [1.0, 7.0]])
        qn1 = kn.get_quantile_norm_matrix(a)

        self.assertEqual(sum(sum(qn1 != aQN)), 0, 'Quantile Norm 1 Not Equal')

    def test_smooth_matrix_with_rwr(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 25
        run_parameters = self.get_run_parameters()
        run_parameters['restart_probability'] = 1
        run_parameters['restart_tolerance'] = 1e-12
        F0 = np.eye(2)
        A = spar.csr_matrix(((np.eye(2) + np.ones(2)) / 3))
        # A = (np.eye(2) + np.ones((2,2))) / 3

        F_exact = np.ones((2, 2)) * 0.5
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS)

        T = (np.abs(F_exact - F_calculated))

        self.assertAlmostEqual(T.sum(), 0)

    def test_smooth_matrix_with_rwr_non_sparse(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 25
        run_parameters = self.get_run_parameters()
        run_parameters['restart_probability'] = 1
        run_parameters['restart_tolerance'] = 1e-12
        F0 = np.eye(2)
        # A = spar.csr_matrix(( (np.eye(2) + np.ones(2)) / 3) )
        A = (np.eye(2) + np.ones((2, 2))) / 3

        F_exact = np.ones((2, 2)) * 0.5
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS)

        T = (np.abs(F_exact - F_calculated))

        self.assertAlmostEqual(T.sum(), 0)

    def test_smooth_matrix_with_rwr_single_vector(self):
        """ Assert that a test matrix will converge to the precomputed answer in
            the predicted number of steps (iterations). Depends on run_parameters
            and the values set herein.
        """
        EXPECTED_STEPS = 25
        run_parameters = self.get_run_parameters()
        run_parameters['restart_probability'] = 1
        run_parameters['restart_tolerance'] = 1e-12
        F0 = np.array([1.0, 0.0])
        # A = spar.csr_matrix(( (np.eye(2) + np.ones(2)) / 3) )
        A = (np.eye(2) + np.ones((2, 2))) / 3

        F_exact = np.array([0.5, 0.5])
        F_calculated, steps = kn.smooth_matrix_with_rwr(F0, A, run_parameters)
        self.assertEqual(steps, EXPECTED_STEPS, msg='minor difference')
        T = (np.abs(F_exact - F_calculated))
        self.assertAlmostEqual(T.sum(), 0)

    def test_normalize_sparse_mat_by_diagonal(self):
        """ assert that a test matrix will be "normalized" s.t. the sum of the rows
            or columns will nearly equal one
        """
        A = np.random.rand(500, 500)
        B = kn.normalize_sparse_mat_by_diagonal(spar.csr_matrix(A))
        B = B.todense()
        B2 = B ** 2
        B2 = np.sqrt(B2.sum())
        geo_mean = float(stats.gmean(B.sum(axis=1)))
        self.assertAlmostEqual(geo_mean, 1, delta=0.1)
        geo_mean = float(stats.gmean(B.T.sum(axis=1)))
        self.assertAlmostEqual(geo_mean, 1, delta=0.1)

    def test_form_network_laplacian_matrix(self):
        """ assert that the laplacian matrix returned sums to zero in both rows
            and columns
        """
        THRESHOLD = 0.8
        A = np.random.rand(10, 10)
        A[A < THRESHOLD] = 0
        A = A + A.T
        Ld, Lk = kn.form_network_laplacian_matrix(A)
        L = Ld - Lk
        L = L.todense()
        L0 = L.sum(axis=0)
        self.assertFalse(L0.any(), msg='Laplacian row sum not equal 0')
        L1 = L.sum(axis=1)
        self.assertFalse(L1.any(), msg='Laplacian col sum not equal 0')
        Ld, Lk = kn.form_network_laplacian_matrix(spar.csr_matrix(A))
        L = Ld - Lk
        L = L.todense()
        L0 = L.sum(axis=0)
        self.assertFalse(L0.any(), msg='Laplacian row sum not equal 0')
        L1 = L.sum(axis=1)
        self.assertFalse(L1.any(), msg='Laplacian col sum not equal 0')

    def test_sample_a_matrix(self):
        """ assert that the random sample is of the propper size, the
            permutation points to the correct columns and that the number of
            rows set to zero is correct.
        """
        n_test_rows = 11
        n_test_cols = 5
        pct_smpl = 0.6
        n_zero_rows = int(np.round(n_test_rows * (1 - pct_smpl)))
        n_smpl_cols = int(np.round(n_test_cols * pct_smpl))
        epsilon_sum = max(n_test_rows, n_test_cols) * 1e-15
        A = np.random.rand(n_test_rows, n_test_cols) + epsilon_sum
        B, P = kn.sample_a_matrix(A, pct_smpl)
        self.assertEqual(B.shape[1], P.size, msg='permutation size not equal columns')
        self.assertEqual(P.size, n_smpl_cols, msg='number of sample columns exception')
        perm_err_sum = 0
        n_zero_err_sum = 0
        B_col = 0
        for A_col in P:
            n_zeros = (np.int_(B[:, B_col] == 0)).sum()
            if n_zeros != n_zero_rows:
                n_zero_err_sum += 1
            C = A[:, A_col] - B[:, B_col]
            C[B[:, B_col] == 0] = 0
            B_col += 1
            if C.sum() > epsilon_sum:
                perm_err_sum += 1

        self.assertEqual(n_zero_err_sum, 0, msg='number of zero columns exception')
        self.assertEqual(perm_err_sum, 0, msg='permutation index exception')

    def test_create_dir_AND_remove_dir(self):
        """ assert that the functions work togeather to create and remove a directory
            even when files have been added
        """
        dir_name = 'tmp_test'
        run_parameters = self.get_run_parameters()
        dir_path = run_parameters['test_directory']
        ndr = kn.create_dir(dir_path, dir_name)
        self.assertTrue(os.path.exists(ndr), msg='create_dir function exception')
        A = np.random.rand(10, 10)
        time_stamp = '123456789'
        a_name = os.path.join(ndr, 'temp_test' + time_stamp)
        with open(a_name, 'wb') as fh:
            A.dump(fh)
        A_back = np.load(a_name)
        if os.path.isfile(a_name):
            os.remove(a_name)
        A_diff = A - A_back
        A_diff = A_diff.sum()
        self.assertEqual(A_diff, 0, msg='write / read directory exception')
        kn.remove_dir(ndr)
        self.assertFalse(os.path.exists(ndr), msg='remove_dir function exception')

    def test_get_timestamp(self):
        """ assert that the default size of the timestamp string is 16 chars and
            that sequential calls produce differnt results
        """
        n_default_chars = 16
        stamp_time = 1e6
        tstr = kn.get_timestamp(stamp_time)
        tstr2 = kn.get_timestamp()

        self.assertEqual(len(tstr), n_default_chars, msg='string return size unexpected')
        self.assertNotEqual(tstr, tstr2)

    def test_create_timestamped_filename(self):
        """ assert that the beginning char string remains unchanged and that the
            size of the returned string is as expected
        """
        n_default_chars = 27
        name_base = 'honky_tonk'
        tsfn = kn.create_timestamped_filename(name_base)
        self.assertEqual(name_base, tsfn[0:10], msg='prefix name exception')
        self.assertEqual(len(tsfn), n_default_chars, msg='filename size exception')

    def test_append_run_parameters_dict(self):
        """ assert that key value pairs are inserted and are retrevable from the run
            parameters dictionary
        """
        run_parameters = self.get_run_parameters()
        run_parameters = kn.append_run_parameters_dict(run_parameters, 'pi_test', np.pi)
        run_parameters = kn.append_run_parameters_dict(run_parameters, 'tea_test', 'tea')

        self.assertEqual(run_parameters['pi_test'], np.pi, msg='float value exception')
        self.assertEqual(run_parameters['tea_test'], 'tea', msg='string value exception')

    def test_update_indicator_matrix(self):
        """ assert that the indicator matrix is not loosing any digits
            Note: correctness test considered as part of linkage matrix test
        """
        n_repeats = 10
        n_test_perm = 11
        n_test_rows = 77
        A = np.zeros((n_test_rows, n_test_rows))
        running_sum = 0
        for r in range(0, n_repeats):
            running_sum += n_test_perm ** 2
            f_perm = np.random.permutation(n_test_rows)
            f_perm = f_perm[0:n_test_perm]
            A = kn.update_indicator_matrix(f_perm, A)

        self.assertEqual(A.sum(), running_sum, msg='sum of elements exception')

    def test_update_linkage_matrix(self):
        """ create a consensus matrix by sampling a synthesized set of clusters
            assert that the clustering is equivalent
        """
        n_samples = 11
        n_clusters = 3
        cluster_set = np.int_(np.ones(n_samples))
        for r in range(0, n_samples):
            cluster_set[r] = int(np.random.randint(n_clusters))

        n_repeats = 33
        n_test_perm = 5
        n_test_rows = n_samples
        I = np.zeros((n_test_rows, n_test_rows))
        M = np.zeros((n_test_rows, n_test_rows))

        for r in range(0, n_repeats):
            f_perm = np.random.permutation(n_test_rows)
            f_perm = f_perm[0:n_test_perm]
            cluster_p = cluster_set[f_perm]
            I = kn.update_indicator_matrix(f_perm, I)
            M = kn.update_linkage_matrix(cluster_p, f_perm, M)

        CC = M / np.maximum(I, 1e-15)

        for s in range(0, n_clusters):
            s_dex = cluster_set == s
            c_c = CC[s_dex, :]
            c_c = c_c[:, s_dex]
            n_check = c_c - 1
            self.assertEqual(n_check.sum(), 0, msg='cluster grouping exception')

            # label_set = kn.perform_kmeans(CC, n_clusters)

    def test_perform_kmeans(self):
        """ assert that the kmeans sets of a known cluster as consensus matrix is the
            same as the known cluster
        """
        n_samples = 11
        n_clusters = 3
        cluster_set = np.int_(np.ones(n_samples))
        for r in range(0, n_samples):
            cluster_set[r] = int(np.random.randint(n_clusters))

        n_repeats = 33
        n_test_perm = 5
        n_test_rows = n_samples
        I = np.zeros((n_test_rows, n_test_rows))
        M = np.zeros((n_test_rows, n_test_rows))

        for r in range(0, n_repeats):
            f_perm = np.random.permutation(n_test_rows)
            f_perm = f_perm[0:n_test_perm]
            cluster_p = cluster_set[f_perm]
            I = kn.update_indicator_matrix(f_perm, I)
            M = kn.update_linkage_matrix(cluster_p, f_perm, M)

        CC = M / np.maximum(I, 1e-15)

        label_set = kn.perform_kmeans(CC, n_clusters)

        self.assertTrue(sets_a_eq_b(cluster_set, label_set), msg='kemans sets differ from cluster')

    def test_update_h_coordinate_matrix(self):
        # epsi_lo = 1e15
        k = 3
        rows = 25
        cols = 6
        W = np.random.rand(rows, k)
        H = np.random.rand(k, cols)
        X = W.dot(H)
        hwx = kn.update_h_coordinate_matrix(W, X)
        dh = (np.abs(H - hwx)).sum()
        self.assertAlmostEqual(dh, 0, msg='h matrix mangled exception')

    def test_perform_nmf(self):
        # assert that nmf finds the same clusters as a known cluster set
        run_parameters = self.get_run_parameters()
        k = run_parameters['k']
        nrows = 90
        ncols = 30

        H0 = np.random.rand(k, ncols)
        C = np.argmax(H0, axis=0)
        H = np.zeros(H0.shape)
        for row in range(0, max(C) + 1):
            rowdex = C == row
            H[row, rowdex] = 1

        W = np.random.rand(nrows, k)
        X = W.dot(H)

        H_b = kn.perform_nmf(X, run_parameters)
        H_clusters = np.argmax(H, axis=0)
        H_b_clusters = np.argmax(H_b, axis=0)

        sets_R_equal = sets_a_eq_b(H_clusters, H_b_clusters)
        self.assertTrue(sets_R_equal, msg='test nmf clusters differ')

    def test_perform_net_nmf(self):
        # assert that net_nmf finds the same clusters as a known cluster set
        # with a basis made from the network
        run_parameters = self.get_run_parameters()
        k = run_parameters['k']
        nrows = 90
        ncols = 30
        H0 = np.random.rand(k, ncols)
        C = np.argmax(H0, axis=0)
        H = np.zeros(H0.shape)
        for row in range(0, max(C) + 1):
            rowdex = C == row
            H[row, rowdex] = 1

        pct_dim = 0.63
        n_nodes = np.int_(np.round(pct_dim * nrows ** 2))
        N = synthesize_random_network(nrows, n_nodes)

        W = np.random.rand(nrows, k)
        W = N.dot(W)

        X = W.dot(H)

        lap_dag, lap_val = kn.form_network_laplacian_matrix(N)
        H_b = kn.perform_net_nmf(X, lap_val, lap_dag, run_parameters)

        H_clusters = np.argmax(H, axis=0)
        H_b_clusters = np.argmax(H_b, axis=0)

        sets_R_equal = sets_a_eq_b(H_clusters, H_b_clusters)
        self.assertTrue(sets_R_equal, msg='test net nmf clusters differ')

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(toolbox_test))

    return test_suite


'''# Next two lines for using this file w/o test Suite   << NOT recommended
#if __name__=='__main__':
#    unittest.main()

                                        >> formal preferred method for using unit test.
import unittest
import TestKEGmodule as tkeg
mySuit = tkn.suite()
runner = unittest.TextTestRunner()
myResult = runner.run(mySuit)

                                        >> OR... One liner: shorter method.
mySuit2 = unittest.TestLoader().loadTestsFromTestCase(TestKEGmodule)

'''
