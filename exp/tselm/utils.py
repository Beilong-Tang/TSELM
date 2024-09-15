"""
This code is used in config/tselm_l.yaml to do some manipulation
"""
from typing import List

def len_( ssl_layers: List[int], vocab_size):
    """
    Return the total vocabularies
    """
    return len(ssl_layers) * vocab_size


def get_len(array: list):
    """
    Return the length of the array
    """
    return len(array)
