#import random
from mario.level_utils import one_hot_to_ascii_level

def platform_test_vec(vec, token_list):
    score = 0.0

    ascii_level = one_hot_to_ascii_level(vec.detach(), token_list)
    for i in range(len(ascii_level[-1])):
        if ascii_level[-1][i] != ascii_level[-2][i]:
            score+=1.0
    
    return score


def platform_test_file(path_to_level_txt):
    n_lines = 0
    is_valid = True
    ascii_levels = []

    with open(path_to_level_txt, "r") as f:
        for line in f:
            n_lines+=1
            ascii_levels.append(line)

    if ascii_levels[-1] != ascii_levels[-2]:
        is_valid = False

    return is_valid

def num_jumps(vec, token_list):
    num_jumps = 0
    ascii_level = one_hot_to_ascii_level(vec.detach(), token_list)
    merged_level = ""
    for i in range(len(ascii_level[-1])):
        if ascii_level[-1][i] == "X" or ascii_level[-2][i] == "X":
            merged_level += "X"
        else:
            merged_level += " "
    
    platforms = merged_level.split()
    return max(len(platforms)-1, 0)