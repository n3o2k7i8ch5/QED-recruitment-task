import math

import numpy as np
import torch
from numpy.core.defchararray import isnumeric


def __build_str_to_int_map():
    all_letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
    ]

    letter_pairs_dict = {}

    i = 0
    for letter_1 in all_letters:
        for letter_2 in all_letters:
            letter_pairs_dict[f'{letter_1}{letter_2}'] = i
            i += 1

    return letter_pairs_dict


__letter_pairs_dict = __build_str_to_int_map()


# there are 26*26 = 676 letter-pairs. We need 10 bits to encode one.
def ip_to_bin_list(ip: str):
    all_bin_arr = []

    if type(ip) != str:
        return [0] * 40

    for part in ip.split('.'):
        if isnumeric(part):
            num_part = int(part)
        else:
            try:
                num_part = __letter_pairs_dict[part]
            except KeyError:
                num_part = 0

        binary = bin(num_part).replace('0b', '')

        bin_arr = [int(b) for b in binary]

        for i in range(10 - len(bin_arr)):
            bin_arr.insert(0, 0)

        all_bin_arr.extend(bin_arr)

    return all_bin_arr


def port_to_bin_list(port):
    if math.isnan(port):
        port = 0
    binary = bin(int(port)).replace('0b', '')
    bin_arr = [int(b) for b in binary]

    for i in range(16 - len(bin_arr)):
        bin_arr.insert(0, 0)

    return bin_arr
