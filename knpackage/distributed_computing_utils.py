'''
    This module provides functions to run code in a parallelized/distributed way
'''


def generate_compute_clusters(cluster_ip_addresses, func_name, dependency_list):
    '''
    Generate clusters based on given list of ip address

    Args:
        cluster_ip_addresses: a list of ip address
        func_name: function name
        dependency_list: the dependencies for running the current function

    Returns:
        cluster_list: a list of clusters as dispy object

    '''
    import sys
    import dispy

    try:
        cluster_list = []
        range_list = range(0, len(cluster_ip_addresses))

        for i in range_list:
            cur_cluster = dispy.JobCluster(func_name,
                                           nodes=[cluster_ip_addresses[i]],
                                           depends=dependency_list,
                                           loglevel=dispy.logger.DEBUG)
            cluster_list.append(cur_cluster)
        return cluster_list
    except:
        print("Unexpected error: {}".format(sys.exc_info()))
        raise


def create_cluster_worker(cluster, i, *args_to_func):
    '''
    Submit a job to cluster.

    Args:
        cluster:
        i:
        *args_to_func: a list of arguments following by the order of arguments defined in
            calling function.

    Returns:

    '''
    import sys

    print("Start creating clusters {}.....".format(str(i)))
    try:
        print("Length of passing arguments = {}".format(len(args_to_func)))
        job = cluster.submit(*args_to_func)
        job.id = i
        ret = job()
        print(ret, job.stdout, job.stderr, job.exception, job.ip_addr, job.start_time, job.end_time)
    except:
        print("Unexpected error: {}".format(sys.exc_info()))
        raise


def parallel_submitting_job_to_each_compute_node(cluster_list, number_of_jobs_each_node, *arguments):
    '''
    Parallel submitting jobs to each node and start computation.

    Args:
        cluster_list:
        number_of_jobs_each_node:
        *arguments: a list of arguments following by the order of arguments defined in
            calling function.

    Returns:

    '''
    import threading
    import sys

    thread_list = []
    received_args = tuple(arguments)
    print("Start spawning {} threads.....".format(len(cluster_list)))
    try:
        for i in range(len(cluster_list)):
            compute_func_args = received_args + (number_of_jobs_each_node[i],)
            t = threading.Thread(target=create_cluster_worker, args=(cluster_list[i], i) + compute_func_args)
            thread_list.append(t)
            t.start()

        for thread in thread_list:
            thread.join()

        for cluster in cluster_list:
            cluster.print_status()

        for cluster in cluster_list:
            cluster.close()
    except:
        print("Unexpected error: {}".format(sys.exc_info()))
        raise


def determine_number_of_compute_nodes(cluster_ip_addresses, number_of_bootstraps):
    '''
    Determine the total number of compute nodes will be used in execution

    Args:
        cluster_ip_addresses: a list of ip address
        number_of_bootstraps:  total number of loops needs to be distributed across clusters

    Returns:
        number_of_compute_nodes: the number of compute nodes

    '''
    available_computing_nodes = len(cluster_ip_addresses)

    if (number_of_bootstraps < available_computing_nodes):
        number_of_compute_nodes = number_of_bootstraps
    else:
        number_of_compute_nodes = available_computing_nodes

    return number_of_compute_nodes


def determine_job_number_on_each_compute_node(number_of_bootstraps, number_of_compute_nodes):
    '''
    Determine total number of jobs run on each compute node

    Args:
        number_of_bootstraps: total number of loops needs to be distributed across compute nodes
        number_of_compute_nodes: total number of available compute nodes

    Returns:
        number_of_scheduled_jobs: a list of integer indicates number of jobs distribution across compute nodes

    '''
    number_of_jobs_on_single_node = int(number_of_bootstraps / number_of_compute_nodes)
    remainder_of_jobs = number_of_bootstraps % number_of_compute_nodes

    number_of_scheduled_jobs = []
    if remainder_of_jobs > 0:
        count = 0
        for i in range(number_of_compute_nodes):
            if (count < remainder_of_jobs):
                number_of_scheduled_jobs.append(number_of_jobs_on_single_node + 1)
            else:
                number_of_scheduled_jobs.append(number_of_jobs_on_single_node)
            count += 1
    else:
        for i in range(number_of_compute_nodes):
            number_of_scheduled_jobs.append(number_of_jobs_on_single_node)

    print("number_of_scheduled_jobs across clusters : {}".format(number_of_scheduled_jobs))
    return number_of_scheduled_jobs


def determine_parallelism_locally(number_of_loops, user_defined_parallelism=0):
    '''
    Determine the parallelism on the current compute node

    Args:
        number_of_loops: total number of loops will be executed on current compute node
        user_defined_parallelism: a customized parallelism specified by users

    Returns:
        number_of_cpu: parallelism on current compute node

    '''
    import multiprocessing

    number_of_cpu = multiprocessing.cpu_count()
    # This condition happens when user_defined_parallelism is defined
    if number_of_loops > 0 and user_defined_parallelism > 0:
        return min(number_of_cpu, number_of_loops, user_defined_parallelism)
    # The following conditions happen when user_defined_parallelism is not defined
    if (number_of_loops <= 0):
        return 1;

    return min(number_of_cpu, number_of_loops)


def move_files(src, dst):
    '''
    Move files from source directory to destination

    Args:
        src: source directory
        dst: destination directory

    Returns:

    '''
    import subprocess
    import sys
    try:
        cmd = ['mv', src, dst]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        print(output)
        print(err)
    except:
        raise OSError(sys.exc_info())


def parallelize_processes_locally(function_name, zipped_arg_list, parallelism):
    '''
    Locally parallelize processes based on resource in local machine

    Args:
        function_name: the worker function to be parallelized
        zipped_arg_list: argument as a zipped object
        number_of_loop_to_be_parallelized: number of loops to be partitioned

    Returns:
        N/A
    '''
    import sys
    import socket
    import multiprocessing

    host = socket.gethostname()
    try:
        p = multiprocessing.Pool(processes=parallelism)
        p.starmap(function_name, zipped_arg_list)

        p.close()
        p.join()
        return "Succeeded in running parallelization on host {}!".format(host)
    except:
        raise OSError("Failed running parallel processing on host {}:{}".format(host, sys.exc_info()))


def zip_parameters(*args):
    '''
    Zip arguments to be an zip object. Note, the last element has to be a range for parallelization

    Args:
        *args: any length of argument with a range to be the last element

    Returns:
        a zipped argument
    '''
    import itertools

    args_list = list(args)
    index_before_last = len(args_list) - 1
    args_list[0:index_before_last] = [itertools.repeat(arg) for arg in args_list[0:index_before_last]]

    return zip(*args_list)


def execute_distribute_computing_job(cluster_ip_address_list, number_of_bootstraps, func_args, dist_main_function,
                                     dependency_list):
    """
    Executes distribute computing job.
    Args:
        cluster_ip_address_list: a list of ip addresses of cluster which will run the job
        number_of_bootstraps: number of bootstraps
        func_args: arguments of the function that will be run in distribute mode
        dist_main_function: the name of the function that will be run in distribute mode
        dependency_list: a list of dependency functions of dist_main_function

    Returns:
        NA
    """
    print("Start distributing jobs......")

    # determine number of compute nodes to use
    number_of_comptue_nodes = determine_number_of_compute_nodes(cluster_ip_address_list, number_of_bootstraps)
    print("Number of compute nodes = {}".format(number_of_comptue_nodes))

    # create clusters
    cluster_list = generate_compute_clusters(
        cluster_ip_address_list[0:number_of_comptue_nodes],
        dist_main_function,
        dependency_list)

    # calculates number of jobs assigned to each compute node
    number_of_jobs_each_node = determine_job_number_on_each_compute_node(number_of_bootstraps, len(cluster_list))

    # parallel submitting jobs
    parallel_submitting_job_to_each_compute_node(cluster_list, number_of_jobs_each_node, *func_args)

    print("Finish distributing jobs......")


