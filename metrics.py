#import random
from mario.level_utils import one_hot_to_ascii_level, ascii_to_one_hot_level
from mario.tokens import ENEMY_TOKENS
from level_snippet_dataset_cmaes import LevelSnippetDataset
import collections, math
from tqdm import tqdm
import numpy as np
import gzip
import sys

def normalized_compression_div(vec, opt):
    ascii_level = one_hot_to_ascii_level(vec.detach(), opt.token_list)

    ref_level = []
    path = opt.input_dir + '/' + opt.input_name

    with open(path, "r") as f:
        for line in f:
            ref_level.append(line) 
    
    x, y = "", ""

    for row in ascii_level:
        x += row

    for row in ref_level:
        y += row

    xy = x + y

    kx = gzip.compress(x.encode('utf-8'))
    ky = gzip.compress(y.encode('utf-8'))
    kxy= gzip.compress(xy.encode('utf-8'))

    ncd = (sys.getsizeof(kxy) - min(sys.getsizeof(kx), sys.getsizeof(ky)))/max(sys.getsizeof(kx), sys.getsizeof(ky))

    return ncd

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

def hamming_dist(vec, opt):
    ascii_level = one_hot_to_ascii_level(vec.detach(), opt.token_list)

    ref_level = []
    path = opt.input_dir + '/' + opt.input_name
    hamming = 0

    with open(path, "r") as f:
        for line in f:
            ref_level.append(line)  

    for i in range(len(ref_level)):
        for j in range(len(ref_level[0])):
            if i== 15 and j==202:
                pass
            else:
                if ascii_level[i][j] != ref_level[i][j]:
                    hamming += 1.0

    hamming_score = hamming/(float(len(ref_level))*float(len(ref_level[0])))

    return hamming_score

def pattern_key(level_slice):
    """
    Computes a hashable key from a level slice.
    """
    key = ""
    for line in level_slice:
        for token in line:
            key += str(token)
    return key


def get_pattern_counts(level, pattern_size):
    """
    Collects counts from all patterns in the level of the given size.
    """
    pattern_counts = collections.defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            down = up + pattern_size
            right = left + pattern_size
            level_slice = level[up:down, left:right]
            pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts


def compute_pattern_counts(dataset, pattern_size):
    """
    Compute pattern counts in parallel from a given dataset.
    """
    levels = [level.argmax(axis = 0) for level in dataset.levels]
    counts_per_level = [get_pattern_counts(level, pattern_size) for level in levels]
    '''
    with mp.Pool() as pool:
        counts_per_level = pool.map(
            partial(get_pattern_counts, pattern_size=pattern_size), levels,
        )
    '''    
    pattern_counts = collections.defaultdict(int)
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    return pattern_counts


def compute_prob(pattern_count, num_patterns, epsilon=1e-7):
    """
    Compute probability of a pattern.
    """
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))


def compute_kl_divergence(level_vector, opt):

    pattern_sizes = [4, 3, 2]
    slice_width = 16
    weight = 1

    ref_level_path = opt.input_dir + '/' + opt.input_name
    ref_level = []

    with open(ref_level_path, "r") as f:
        for line in f:
            ref_level.append(line)

    ref_level_vector = ascii_to_one_hot_level(ref_level, opt.token_list)

    ref_dataset = LevelSnippetDataset(
        level_vector=ref_level_vector,
        slice_width=slice_width,
        token_list=opt.token_list,
        ascii_level = ref_level
    )
    
    test_dataset = LevelSnippetDataset(
        level_vector=level_vector,
        slice_width=slice_width,
        token_list=opt.token_list,
    )

    kl_divergences = []
    for pattern_size in pattern_sizes:
        pattern_counts = compute_pattern_counts(ref_dataset, pattern_size)
        test_pattern_counts = compute_pattern_counts(
            test_dataset, pattern_size)

        num_patterns = sum(pattern_counts.values())
        num_test_patterns = sum(test_pattern_counts.values())

        kl_divergence = 0
        for pattern, count in tqdm(pattern_counts.items()):
            prob_p = compute_prob(count, num_patterns)
            prob_q = compute_prob(
                test_pattern_counts[pattern], num_test_patterns)
            kl_divergence += weight * prob_p * math.log(prob_p / prob_q) + (
                1 - weight
            ) * prob_q * math.log(prob_q / prob_p)

        kl_divergences.append(kl_divergence)

    mean_kl_divergence = np.mean(kl_divergences)
    var_kl_divergence = np.std(kl_divergences)
    return mean_kl_divergence, var_kl_divergence
