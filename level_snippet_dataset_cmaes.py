import math
import os

import torch
from torch.utils.data import Dataset

from level_utils import load_level_from_text, ascii_to_one_hot_level, one_hot_to_ascii_level


class LevelSnippetDataset(Dataset):
    """
    Converts a folder (level_dir) with token based ascii-levels in .txt files into a torch Dataset of slice_width by
     slice_width level slices. Default for Super Mario Bros. is 16, as the levels are 16 pixels high. level_dir needs
     to only include level.txt files.

     token_list : If None, token_list is calculated internally. Can be set for different future applications.
     level_idx : If None, __getitem__ returns the actual index with the retrieved slice. Can be set for different future
        applications
    level_name : If None, all level files in folder are used, otherwise only level_name will be used.
    """
    def __init__(self, level_vector, ascii_level = None, slice_width=16, token_list=None, level_idx=None, level_name=None,):
        super(LevelSnippetDataset, self).__init__()
        self.level_idx = level_idx
        self.ascii_levels = []
        self.levels = []
        uniques = set()
        if ascii_level == None:
            ascii_level = one_hot_to_ascii_level(level_vector, token_list)
        for line in ascii_level:
            for token in line:
                if token != "\n" and token != "M" and token != "F":
                    uniques.add(token)
        self.ascii_levels.append(ascii_level)

        curr_level = ascii_to_one_hot_level(ascii_level, token_list)
        self.levels.append(curr_level)

        self.slice_width = slice_width
        self.missing_slices_per_level = slice_width - 1
        self.missing_slices_l = math.floor(self.missing_slices_per_level / 2)
        self.missing_slices_r = math.ceil(self.missing_slices_per_level / 2)

        self.level_lengths = [
            x.shape[-1] - self.missing_slices_per_level for x in self.levels
        ]


    def __getitem__(self, idx):
        i_l = 0
        while sum(self.level_lengths[0:i_l]) < (idx + 1) < sum(self.level_lengths):
            i_l += 1
        i_l -= 1

        level = self.levels[i_l]
        idx_lev = idx - sum(self.level_lengths[0:i_l]) + self.missing_slices_l
        lev_slice = level[
            :, :, idx_lev - self.missing_slices_l: idx_lev + self.missing_slices_r + 1
        ]
        return (
            lev_slice,
            torch.tensor(i_l if self.level_idx is None else self.level_idx),
        )

    def __len__(self):
        return sum(self.level_lengths) - 1
