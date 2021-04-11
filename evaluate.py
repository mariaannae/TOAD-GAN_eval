def platform_test(path_to_level_txt):
    n_lines = 0
    is_valid = True
    ascii_levels = []

    with open(path_to_level_txt, "r") as f:
        for line in f:
            n_lines+=1
            ascii_levels.append(line)
    
    for lev in ascii_levels[-2:]:
        print(lev)
    
    if ascii_levels[-1] != ascii_levels[-2]:
        is_valid = False

    return is_valid

print(platform_test('./input/lvl_1-1.txt'))