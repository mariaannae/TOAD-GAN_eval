import random
import os

import numpy as np
import torch
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageDraw
from level_utils import read_level_from_file, one_hot_to_ascii_level, place_a_mario_token, ascii_to_one_hot_level
from level_image_gen import LevelImageGen

# Define Variables
#root = Tk(className=" TOAD-GUI")







# Set values

#load_string_gen.set("Click the buttons to open a level or generator.")
#load_string_txt.set(os.path.join(os.curdir, "levels"))


# Path to the AI Framework jar for Playing levels
# MARIO_AI_PATH = os.path.abspath(os.path.join(os.path.curdir, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar"))

# Level to Image renderer


# Object holding important data about the current level
class LevelObject:
    def __init__(self, ascii_level, oh_level, image, tokens, scales, noises):
        self.ascii_level = ascii_level
        self.oh_level = oh_level  # one-hot encoded
        self.image = image
        self.tokens = tokens
        self.scales = scales
        self.noises = noises


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

