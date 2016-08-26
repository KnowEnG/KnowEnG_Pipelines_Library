# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:39:35 2016

@author: The Gene Sets Characterization dev team

"""
import os
import argparse
import time
import numpy as np
import numpy.linalg as LA
from numpy import maximum

import pandas as pd
import scipy.sparse as spar
from scipy.sparse import csgraph
from sklearn.cluster import KMeans

import yaml

def get_run_directory_and_file(args):
    """ Read system input arguments (argv) to get the run directory name.

    Args:
        args: sys.argv, command line input; python main -run_directory dir_name

    Returns:
        run_directory: directory where run_file is expected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    parser.add_argument('-run_file', type=str)
    args = parser.parse_args()
    run_directory = args.run_directory
    run_file = args.run_file

    return run_directory, run_file

def get_run_parameters(run_directory, run_file):
    """ Read system input arguments run directory name and run_file into a dictionary.

    Args:
        run_directory: directory where run_file is expected.

    Returns:
        run_parameters: python dictionary of name - value parameters.
    """
    run_file_name = os.path.join(run_directory, run_file)
    with open(run_file_name, 'r') as file_handle:
        run_parameters = yaml.load(file_handle)

    run_parameters["run_directory"] = run_directory
    run_parameters["run_file"] = run_file

    return run_parameters


def get_spreadsheet_df(spreadsheet_name_full_path):
    """ get the spreadsheet file name from the run_parameters dictionary and
        read the file into a pandas dataframe.

    Args:
        spreadsheet_name_full_path: full path name of a tab separated values spreadsheet
            with row and column names

    Returns:
        spreadsheet_df: the spreadsheet dataframe.
    """

    spreadsheet_df = pd.read_csv(
        spreadsheet_name_full_path, sep='\t', header=0, index_col=0)

    return spreadsheet_df

def get_network_df(network_name):
    """ Read in the cleaned subnet from KnowEnG network.

    Args:
        network_name: file name of cleaned network

    Returns:
        network_df: 3-column dataframe of cleaned network
    """
    try:
        f = open(network_name)
    except IOError:
        print('cannot open', network_name)
    else:
        f.close()

    network_df = pd.read_csv(
        network_name, header=None, names=None, delimiter='\t', usecols=[0, 1, 2])
    network_df.columns = ['node_1', 'node_2', 'wt']

    return network_df

def extract_network_node_names(network_df):
    """ extract node names lists from network.

    Args:
        netwrok_df: network dataframe.

    Returns:
        node_1_names: all names in column 1.
        node_list_2: all names in column 2.
    """


    node_list_1 = list(set(network_df.values[:, 0]))
    node_list_2 = list(set(network_df.values[:, 1]))

    return node_list_1, node_list_2

def find_unique_node_names(node_list_1, node_list_2):
    """ get the list (set union) of genes in either of the input lists.

    Args:
        node_list_1: list of node names.
        node_list_2: list of node names.

    Returns:
        unique_node_names: unique list of all node names.
    """
    unique_node_names = list(set(node_list_1) | set(node_list_2))

    return unique_node_names

def find_common_node_names(node_list_1, node_list_2):
    """ get the list (set intersection) of genes in both of the input lists.

    Args:
        node_list_1: list of node names.
        node_list_2: list of node names.

    Returns:
        common_node_names: unique list of common node names.
    """
    common_node_names = list(set(node_list_1) & set(node_list_2))

    return common_node_names

def extract_spreadsheet_gene_names(spreadsheet_df):
    """ get the uinque list (df.index.values) of genes in the spreadsheet dataframe.

    Args:
        spreadsheet_df: dataframe of spreadsheet input file.

    Returns:
        spreadsheet_gene_names: list of spreadsheet genes.
    """
    spreadsheet_gene_names = list(set(spreadsheet_df.index.values))

    return spreadsheet_gene_names

def find_dropped_node_names(spreadsheet_df, unique_gene_names):
    """ write list of genes dropped from the input spreadsheed to
        run_parameters['tmp_directory'].file_name.

    Args:
        spreadsheet_df: the full spreadsheet data frame before dropping.
        unique_gene_names: the genes that will be used in calculation.
    """
    droplist = spreadsheet_df.loc[~spreadsheet_df.index.isin(unique_gene_names)]
    droplist = droplist.index.values
    return droplist

def update_spreadsheet_df(spreadsheet_df, gene_names):
    """ resize and reorder spreadsheet dataframe to only the gene_names list.

    Args:
        spreadsheet_df: dataframe of spreadsheet.
        unique_gene_names: list of all genes in network.

    Returns:
        pandas dataframe of spreadsheet with only network genes.
    """

    return spreadsheet_df.loc[gene_names].fillna(0)

def update_network_df(network, nodes_list, node_id):
    """ remove nodes not found as nodes_list in network node_id.

    Args:
        network: property to gene edges.
        intersection: user provided dataframe.

    Returns:
        network that contains (rows) nodes_list found in node_id.
    """

    return network[network[node_id].isin(nodes_list)]

def create_node_names_dict(node_names, start_value=0):
    """ create a python dictionary to look up gene locations from gene names

    Args:
        unique_gene_names: python list of gene names

    Returns:
        node_names_dictionary: python dictionary of gene names to integer locations
    """
    index_length = len(list(node_names)) + start_value
    node_names_dictionary = dict(zip(node_names, np.arange(start_value, index_length)))

    return node_names_dictionary

def create_reverse_node_names_dict(dictionary):
    """ create reverse dictionary (keys > values, values > keys).

    Args:
        dictionary: dictionary.

    Returns:
        reverse dictionary: dictionary.
    """

    return {value: key for key, value in dictionary.items()}

def symmetrize_df(network):
    """ symmetrize network in sparse (3 cloumn) form.
        Note: expecting zero or no diagonal.

    Args:
        network: property to gene edges (with nothing on the diagonal).

    Returns:
        symm_network: symm_network[r, c] == symm_network[c, r], (network extended).
    """
    # if not network:
    #     print('empty network')
    #     return False
    #
    # if list(network.columns.values)!=['node_1', 'node_2', 'wt']:
    #     print('wrong format, need to change column names')
    #     return False

    transpose = pd.DataFrame()
    transpose['node_1'] = network['node_2']
    transpose['node_2'] = network['node_1']
    transpose['wt'] = network['wt']
    symm_network = pd.concat([network, transpose], ignore_index=True)

    return symm_network

def map_node_names_to_index(network_df, genes_map, node_id):
    """ replace the node names with numbers for formation of numeric sparse matrix.

    Args:
        network_df: 3 col data frame version of network.
        genes_lookup_table: genes to location index dictionary.

    Returns:
        network_df: the same dataframe with integer indices in columns 0, 1.
    """
    # if node_id not in network_df:
    #     print("invalid node_id")
    #     return False

    network_df[node_id] = [genes_map[i] for i in network_df[node_id]]

    return network_df

def create_df_with_sample_labels(sample_names, labels):
    """ create dataframe from spreadsheet column names with cluster number assignments.

    Args:
        sample_names: spreadsheet column names.
        labels: cluster number assignments.

    Returns:
        clusters_dataframe: dataframe with sample_names keys to labels values.
    """
    clusters_dataframe = pd.DataFrame(data=labels, index=sample_names)

    return clusters_dataframe

def convert_network_df_to_sparse(pg_network_df, row_size, col_size):
    """ convert global network to sparse matrix.

    Args:
        pg_network_df: property-gene dataframe of global network (3 col)
        row_size: number of rows in sparse outpu
        col_size: number of columns in sparse outpu

    Returns:
        pg_network_sparse: sparse matrix of network gene set.
    """
    # if type(row_size)!=int or type(col_size)!=int:
    #     print('Wrong sparse matrix size data type')
    #     return
    # if pg_network_df.empty:
    #     print("Empty input network")
    #     return

    row_iden = pg_network_df.values[:, 1]
    col_iden = pg_network_df.values[:, 0]

    # if row_size-1<max(list(row_iden)) or col_size-1<max(list(col_iden)):
    #     print('Size does not match')
    #     return

    data = pg_network_df.values[:, 2]
    pg_network_sparse = spar.csr_matrix(
        (data, (row_iden, col_iden)), shape=(row_size, col_size))

    return pg_network_sparse

def save_df(result_df, tmp_dir, file_name):
    """ save the result of DRaWR in tmp directory, file_name.

    Args:
        rw_result_df: dataframe of random walk result.
        tmp_dir: directory to save the result file.
        file_name: file name to save to.
    """
    # if not os.path.isdir(tmp_dir):
    #     print('invalid tmp_dir')
    #     return False
        
    file_path = os.path.join(tmp_dir, file_name)
    result_df.to_csv(file_path, header=True, index=False, sep='\t')

def append_column_to_spreadsheet(spreadsheet_df, column, col_name):
    """ append baseline vector of the user spreadsheet matrix.

    Args:
        spreadsheet_df: user spreadsheet dataframe.
        column: the column to append, length = spreadsheet_df.shape[0]
        col_name: the column name for the appended column
    Returns:w
        spreadsheet_df: new dataframe with baseline vector appended in the last column.
    """
    # if col_name not in spreadsheet_df:
    #     print('invalid col_name')
    #     return False
    # if not column:
    #     print('empty column input')
    #     return False
    spreadsheet_df[col_name] = column

    return spreadsheet_df

def normalize_network_df_by_sum(network_df, node_id):
    """ normalize the network column with numbers for input.
        Note: expecting zero or no diagonal.

    Args:
        network_df: network dataframe (with nothing on the diagonal).
        node_id: column name

    Returns:
        network_df: the same dataframe with weight normalized.
    """
    # if node_id not in network_df:
    #     print('invalid node_id')
    #     return False
    network_df[node_id] /= network_df[node_id].sum()

    return network_df

def form_hybrid_network_df(list_of_networks):
    """ concatenate a list of networks.
        Note: expecting zero or no diagonal.

    Args:
        list_of_networks: a list of networks to join (with nothing on the diagonal).

    Returns:
        a combined hybrid network
    """
    # if not list_of_networks:
    #     print('empty input')
    #     return False
    return pd.concat(list_of_networks, ignore_index=True)

def normalize_sparse_mat_by_diagonal(network_mat):
    """ square root of inverse of diagonal D (D * network_mat * D) normaization.

    Args:
        network_mat: symmetric matrix.

    Returns:
        network_mat: renomralized such that the sum of any row or col is about 1.
    """
    row_sm = np.array(network_mat.sum(axis=0))
    row_sm = 1.0 / row_sm
    row_sm = np.sqrt(row_sm)
    r_c = np.arange(0, network_mat.shape[0])
    diag_mat = spar.csr_matrix((row_sm[0, :], (r_c, r_c)), shape=(network_mat.shape))
    network_mat = diag_mat.dot(network_mat)
    network_mat = network_mat.dot(diag_mat)

    return network_mat

def form_network_laplacian_matrix(network_mat):
    """ get the laplacian matrix components needed by net nmf

    Args:
        network_mat: sparse symmetric matrix

    Returns:
        diag_laplacian: The diagonal component of the Laplacian matrix
        locs_laplacian: The values component of the Laplacian. L = D - V
    """
    n_mat = network_mat.copy()
    n_mat[n_mat != 0] = 1
    locs_laplacian, diag_laplacian = csgraph.laplacian(n_mat, normed=False, return_diag=True)
    diag_laplacian = spar.diags(diag_laplacian)
    locs_laplacian = spar.csr_matrix(diag_laplacian - locs_laplacian)

    return diag_laplacian, locs_laplacian

def sample_a_matrix(spreadsheet_mat, rows_fraction, cols_fraction):
    """ percent_sample x percent_sample random sample, from spreadsheet_mat.

    Args:
        spreadsheet_mat: gene x sample spread sheet as matrix.
        percent_sample: decimal fraction (slang-percent) - [0 : 1].

    Returns:
        sample_random: A specified precentage sample of the spread sheet.
        sample_permutation: the array that correponds to columns sample.
    """
    features_size = int(np.round(spreadsheet_mat.shape[0] * (1-rows_fraction)))
    features_permutation = np.random.permutation(spreadsheet_mat.shape[0])
    features_permutation = features_permutation[0:features_size].T

    patients_size = int(np.round(spreadsheet_mat.shape[1] * cols_fraction))
    sample_permutation = np.random.permutation(spreadsheet_mat.shape[1])
    sample_permutation = sample_permutation[0:patients_size]

    sample_random = spreadsheet_mat[:, sample_permutation]
    sample_random[features_permutation[:, None], :] = 0

    positive_col_set = sum(sample_random) > 0
    sample_random = sample_random[:, positive_col_set]
    sample_permutation = sample_permutation[positive_col_set]

    return sample_random, sample_permutation

def smooth_matrix_with_rwr(restart, network_sparse, run_parameters):
    """ simulate a random walk with restart. iterate: (R_n+1 = a*N*R_n + (1-a)*R_n).

    Args:
        restart: restart array of any column size.
        network_sparse: network stored in sparse format.
        run_parameters: parameters dictionary with "rwr_restart_probability",
        "rwr_convergence_tolerence", "rwr_max_iterations".

    Returns:
        smooth_1: smoothed restart data.
        step: number of iterations (converged to tolerence or quit).
    """
    tol = np.float_(run_parameters["rwr_convergence_tolerence"])
    alpha = np.float_(run_parameters["rwr_restart_probability"])
    smooth_0 = restart
    smooth_r = (1. - alpha) * restart
    for step in range(0, int(run_parameters["rwr_max_iterations"])):
        smooth_1 = alpha * network_sparse.dot(smooth_0) + smooth_r
        deltav = LA.norm(smooth_1 - smooth_0)
        if deltav < tol:
            break
        smooth_0 = smooth_1

    return smooth_1, step

def get_quantile_norm_matrix(sample):
    """ normalizes an array using quantile normalization (ranking).

    Args:
        sample: initial sample - spreadsheet matrix.

    Returns:
        sample_quantile_norm: quantile normalized spreadsheet matrix.
    """
    index = np.argsort(sample, axis=0)
    sample_sorted_by_rows = np.sort(sample, axis=0)
    mean_per_row = sample_sorted_by_rows.mean(1)
    sample_quantile_norm = sample.copy()
    for j in range(0, sample.shape[1]):
        sample_quantile_norm[index[:, j], j] = mean_per_row[:]

    return sample_quantile_norm

def update_h_coordinate_matrix(w_matrix, x_matrix):
    """ nonnegative right factor matrix for perform_net_nmf function s.t. X ~ W.H.

    Args:
        w_matrix: the positive left factor (W) of the perform_net_nmf function.
        x_matrix: the postive matrix (X) to be decomposed.

    Returns:
        h_matrix: nonnegative right factor (H) matrix.
    """
    wtw = np.dot(w_matrix.T, w_matrix)
    number_of_clusters = wtw.shape[0]
    wtx = np.dot(w_matrix.T, x_matrix)
    colix = np.arange(0, x_matrix.shape[1])
    rowix = np.arange(0, w_matrix.shape[1])
    h_matrix = np.dot(LA.pinv(wtw), wtx)
    h_pos = h_matrix > 0
    h_matrix[~h_pos] = 0
    col_log_arr = sum(h_pos == 0) > 0
    col_list = colix[col_log_arr]
    for cluster in range(0, number_of_clusters):
        if col_list.size > 0:
            w_ette = wtx[:, col_list]
            m_rows = w_ette.shape[0]
            n_cols = w_ette.shape[1]
            mcode_uniq_col_ix = np.arange(0, n_cols)
            h_ette = np.zeros((m_rows, n_cols))
            h_pos_ette = h_pos[:, col_list]
            mcoding = np.dot(2**(np.arange(0, m_rows)), np.int_(h_pos_ette))
            mcode_uniq = np.unique(mcoding)
            for u_n in mcode_uniq:
                ixidx = mcoding == u_n
                c_pat = mcode_uniq_col_ix[ixidx]
                if c_pat.size > 0:
                    r_pat = rowix[h_pos_ette[:, c_pat[0]]]
                    atmp = wtw[r_pat[:, None], r_pat]
                    btmp = w_ette[r_pat[:, None], c_pat]
                    atmptatmp = np.dot(atmp.T, atmp)
                    atmptatmp = LA.pinv(atmptatmp)
                    atmptbtmp = np.dot(atmp.T, btmp)
                    h_ette[r_pat[:, None], c_pat] = np.dot(atmptatmp, atmptbtmp)
                    h_matrix[:, col_list] = h_ette
            h_pos = h_matrix > 0
            h_matrix[~h_pos] = 0
            col_log_arr = sum(h_pos == 0) > 0
            col_list = colix[col_log_arr]
        else:
            break

    return h_matrix

def perform_net_nmf(x_matrix, lap_val, lap_dag, run_parameters):
    """ perform network based nonnegative matrix factorization, minimize:
        ||X-WH|| + lambda.tr(W'.L.W), with W, H positive.

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W.H
        lap_val: the laplacian matrix
        lap_dag: the diagonal of the laplacian matrix
        run_parameters: parameters dictionary with keys: "k", "lambda", "nmf_max_iterations",
            "nmf_max_invariance", "nmf_conv_check_freq".

    Returns:
        h_matrix: nonnegative right factor (H) matrix.
    """
    k = int(run_parameters["number_of_clusters"])
    nmf_penalty_parameter = float(run_parameters["nmf_penalty_parameter"])
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, int(run_parameters["nmf_max_iterations"])):
        if np.mod(itr, int(run_parameters["nmf_conv_check_freq"])) == 0:
            h_clusters = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(h_clust_eq != h_clusters) == 0):
                h_eq_count = h_eq_count + int(run_parameters["nmf_conv_check_freq"])
            else:
                h_eq_count = 0
            h_clust_eq = h_clusters
            if h_eq_count >= float(run_parameters["nmf_max_invariance"]):
                break
        numerator = maximum(np.dot(x_matrix, h_matrix.T) + nmf_penalty_parameter * lap_val.dot(w_matrix), epsilon)
        denomerator = maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T))
                              + nmf_penalty_parameter * lap_dag.dot(w_matrix), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = update_h_coordinate_matrix(w_matrix, x_matrix)

    return h_matrix

def perform_nmf(x_matrix, run_parameters):
    """ nonnegative matrix factorization, minimize the diffence between X and W dot H
        with positive factor matrices W, and H.

    Args:
        x_matrix: the postive matrix (X) to be decomposed into W dot H.
        run_parameters: parameters dictionary with keys "k", "nmf_max_iterations",
            "cluster_min_repeats", "nmf_conv_check_freq".

    Returns:
        h_matrix: nonnegative right factor matrix (H).
    """
    k = int(run_parameters["number_of_clusters"])
    nmf_conv_check_freq = int(run_parameters["nmf_conv_check_freq"])
    nmf_max_invariance = float(run_parameters["nmf_max_invariance"])
    epsilon = 1e-15
    w_matrix = np.random.rand(x_matrix.shape[0], k)
    w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
    h_matrix = np.random.rand(k, x_matrix.shape[1])
    h_clust_eq = np.argmax(h_matrix, 0)
    h_eq_count = 0
    for itr in range(0, int(run_parameters["nmf_max_iterations"])):
        if np.mod(itr, nmf_conv_check_freq) == 0:
            h_clusters = np.argmax(h_matrix, 0)
            if (itr > 0) & (sum(h_clust_eq != h_clusters) == 0):
                h_eq_count = h_eq_count + nmf_conv_check_freq
            else:
                h_eq_count = 0
            h_clust_eq = h_clusters
            if h_eq_count >= nmf_max_invariance:
                break
        numerator = maximum(np.dot(x_matrix, h_matrix.T), epsilon)
        denomerator = maximum(np.dot(w_matrix, np.dot(h_matrix, h_matrix.T)), epsilon)
        w_matrix = w_matrix * (numerator / denomerator)
        w_matrix = maximum(w_matrix / maximum(sum(w_matrix), epsilon), epsilon)
        h_matrix = update_h_coordinate_matrix(w_matrix, x_matrix)

    return h_matrix

def update_linkage_matrix(encode_mat, sample_perm, linkage_matrix):
    ''' update the connectivity matrix by summing the un-permuted linkages.
    encode_mat: (permuted) nonnegative right factor matrix (H) - encoded linkage.
    Args:
        encode_mat: encoding of linkage either as an h_matrix or argmax(h_matrix)

        sample_perm: the sample permutaion of the h_matrix.
        linkage_matrix: connectivity matrix.

    Returns:
        linkage_matrix: connectivity matrix summed with the de-permuted linkage.
    '''
    if encode_mat.ndim == 1:
        num_clusters = max(encode_mat) + 1
        cluster_id = encode_mat
    else:
        num_clusters = encode_mat.shape[0]
        cluster_id = np.argmax(encode_mat, 0)

    for cluster in range(0, num_clusters):
        slice_id = sample_perm[cluster_id == cluster]
        linkage_matrix[slice_id[:, None], slice_id] += 1

    return linkage_matrix

def update_indicator_matrix(sample_perm, indicator_matrix):
    ''' update the indicator matrix by summing the un-permutation.

    Args:
        sample_perm: permutaion of the sample (h_matrix).
        indicator_matrix: indicator matrix.

    Returns:
        indicator_matrix: indicator matrix incremented at sample_perm locations.
    '''
    indicator_matrix[sample_perm[:, None], sample_perm] += 1

    return indicator_matrix

def perform_kmeans(consensus_matrix, k=3):
    """ determine cluster assignments for consensus matrix using K-means.

    Args:
        consensus_matrix: connectivity / indicator matrix.
        k: clusters estimate.

    Returns:
        lablels: ordered cluster assignments for consensus_matrix (samples).
    """
    cluster_handle = KMeans(k, random_state=10)
    labels = cluster_handle.fit_predict(consensus_matrix)

    return labels

def get_timestamp(stamp_units=1e6):
    """ get a time stamp string - current time as integer string.

    Args:
        stamp_units: inverse of time resolution 1e6 returns microseconds.

    Returns:
        timestamp_string: a string of integer digits.
    """
    timestamp_string = np.str_(int(time.time() * np.maximum(stamp_units, 1)))

    return timestamp_string

def create_timestamped_filename(name_base, name_extension=None, precision=None, n_digits=9):
    """ insert a time stamp into the filename_ before .extension.

    Args:
        name_base: file name first part - may include directory path.
        name_extension: file extension without a period.
        precision: None or a multiplier to move the decimal point in seconds
        n_digits: (positive integer) number of decimal digits

    Returns:
        time_stamped_file_name: concatenation of time-stamp between name_base and name_extension.
    """
    dt_min = 1
    if precision is None:
        t0 = time.time()
        t_dec = t0 - np.floor(t0)
        t_dec = '{}'.format(t_dec)
        nstr = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()) + t_dec[1:max(1, n_digits+2)]
    else:
        time_step = max(dt_min, precision)
        nstr = np.str_(int(time.time() * time_step))

    if name_extension is None:
        time_stamped_file_name = name_base + '_' + nstr
    else:
        time_stamped_file_name = name_base + '_' + nstr + '.' + name_extension

    return time_stamped_file_name

def append_run_parameters_dict(run_parameters, key_name, value_str):
    """ add a key-value pair to the run parameters dictionary.

    Args:
        run_parameters: dictionary to append.
        key_name: key name to add or overwrite.
        value_str: value to insert in run_parameters[key_name].

    Returns:
        run_parameters: dictionary with new (or overwritten) key value pair.
    """
    if not value_str:
        print('empty value_str')
        return False
    if not key_name:
        print('empty key_name')
        return False
    run_parameters[key_name] = value_str

    return run_parameters

def create_dir(dir_path, dir_name, timestamp=None):
    """ create a "dir_name" with time stamp directory

    Args:
        dir_name: an existing directory such as the run directory.
        timestamp: optional - if not input a microsecond stamp will be added.
    Returns:
        new_dir_name:
    """
    if timestamp is None:
        timestamp = get_timestamp()

    new_dir_name = os.path.join(dir_path, dir_name + timestamp)
    os.mkdir(new_dir_name)

    return new_dir_name

def remove_dir(dir_name):
    """ remove directory and all the files it contains.

    Args:
        dir_name: name of a directory with no sub-directories.
    """
    dir_list = os.listdir(dir_name)
    if len(dir_list) > 0:
        for file_name in dir_list:
            os.remove(os.path.join(dir_name, file_name))

    os.rmdir(dir_name)

    return
