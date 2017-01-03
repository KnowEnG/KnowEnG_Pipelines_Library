import numpy as np
import unittest
from unittest import TestCase

import knpackage.toolbox as kn

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

class TestPerform_kmeans(TestCase):

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

if __name__ == '__main__':
    unittest.main()