import os.path
from typing import List


def getname_(dir_name, path_name):
    """
    This function return a string using the base_name of path_name and join it with the dir_name
    """
    return os.path.join(dir_name, os.path.basename(path_name))


def len_( ssl_layers: List[int], vocab_size):
    return len(ssl_layers) * vocab_size


def get_len(array: list):
    return len(array)
