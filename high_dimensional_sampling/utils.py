import time
import datetime
import os
import hashlib
import numpy as np


def get_time():
    """
    Returns the current time in milliseconds

    Returns:
        Integer representing the current time in milliseconds
    """
    return int(round(time.time() * 1000))

def get_datetime():
    """
    Returns a string containing the current date and time

    Returns:
        String containign the current date and time
    """
    return str(datetime.datetime.now())

def create_unique_folder(path, prefered):
    """
    Creates a folder with a unique name at the specified path.

    Creates a folder with a name specified by the user at the specified path.
    If this folder already exists, the name of the folder is appended by
    an underscore and the lowest >0 integer number that fixes the naming
    conflict.

    Args:
        path: Path at which the folder should be created
        prefered_name: Name of the folder that should be created
    """
    # Strip folder separator from path if it ends with one
    if path[-1] == os.sep:
        path = path[:-1]
    # Attempt create folder
    folder_path = path + os.sep + prefered
    folder_created = False
    i = 0
    while not folder_created:
        if not os.path.exists(folder_path):
            # Folder does not exist, create it and stop the while loop
            os.makedirs(folder_path)
            folder_created = True
        else:
            # Folder already exists, append _# to the name and try again
            i += 1
            folder_path = path + os.sep + prefered + '_' + str(i)
    return folder_path
        
def benchmark_matrix_inverse():
    """
    Benchmark the user's setup by measuring the time taken by matrix inversion

    Performs a benchmark of the user's setup by inverting a 6400x6400 matrix
    filled with random numbers. This function then returns the time in ms taken
    for this operation.

    Good performance on this benchmark indicates both the CPU and RAM are fast.

    Returns:
        Integer representing the time in milliseconds taken by the matrix
        creation, inversion and deletion.
    """
    t_start = get_time()
    x = np.random.rand(6400, 6400)
    x = np.linalg.inv(x)
    del(x)
    return get_time() - t_start

def benchmark_sha_hashing():
    """
    Benchmark the user's setup by measuring the time taken by hashing

    Performs a benchmark of the user's setup by taking a fixed string and
    perform a sha1-hashing operation 1000000 times on it. When finished, the
    time needed for these operations is returned (in milliseconds).

    Good performance on this benchmark indicates a fast CPU.

    Returns:
        Integer representing the time in milliseconds taken by the million
        sha1-hashing operations.
    """
    t_start = get_time()
    s = b'Let\'s hash some stuff'
    for _ in range(1000000):
        o = hashlib.sha1(s)
        s = bytes(o.hexdigest(), 'utf-8')
    return get_time() - t_start