import time
import datetime
import os
import hashlib
import numpy as np


def get_time():
    return int(round(time.time() * 1000))

def get_datetime():
    return datetime.datetime.now()

def create_unique_folder(path, prefered):
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
    # Return path to created folder
    return folder_path
        
def benchmark_matrix_inverse():
    t_start = get_time()
    x = np.random.rand(6400, 6400)
    x = np.linalg.inv(x)
    del(x)
    return get_time() - t_start

def benchmark_sha_hashing():
    t_start = get_time()
    s = b'Let\'s hash some stuff'
    for i in range(1000000):
        o = hashlib.sha1(s)
        s = bytes(o.hexdigest(), 'utf-8')
    return get_time() - t_start