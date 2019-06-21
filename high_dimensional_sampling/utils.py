import time
import os


def get_time():
    return int(round(time.time() * 1000))

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
        
