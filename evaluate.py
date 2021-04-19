
def platform_test_vec(vec):
    platform1 = vec[:,:,-1,:]
    platform2 = vec[:,:,-2,:]
    mismatch = (platform1 != platform2).sum()
    if mismatch > 0:
        return 0
    else: return 1

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

