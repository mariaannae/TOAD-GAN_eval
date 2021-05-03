import os
import time
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas
from torch.utils.tensorboard import SummaryWriter

from config import get_arguments, post_config
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from generate_noise import generate_spatial_noise
from generate_samples_cmaes import generate_samples_cmaes
from models import load_trained_pyramid

from metrics import platform_test_vec, num_jumps
from random_network import create_random_network
from playability import test_playability

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap


if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    # Parse arguments
    parse = get_arguments()
    parse.add_argument("--out_", help="folder containing generator files", default="output/wandb/latest-run/files/")
    parse.add_argument("--scale_v", type=float, help="vertical scale factor", default=1.0)
    parse.add_argument("--scale_h", type=float, help="horizontal scale factor", default=1.0)
    parse.add_argument("--gen_start_scale", type=int, help="scale to start generating in", default=0)
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=10)
    parse.add_argument("--multiproc", action="store_true", help="run with multiprocessing", default=False)
    parse.add_argument("--experiment_id", type=int, help="the experiment number to load from", default = 1)

    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
            parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)
    
    #setting number of samples generated to 1, so that each noise vector produced corresponds to only one generated level and is not divided into multiple levels
    opt.num_samples = 1

    token_insertion = False

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'mario':
        opt.ImgGen = MarioLevelGen(sprite_path)
        replace_tokens = MARIO_REPLACE_TOKENS
        downsample = special_mario_downsampling

    else:
        NameError("name of --game not recognized. Supported: mario")

    # Load level
    real = read_level(opt, None, replace_tokens).to(opt.device)
    # Load Generator
    generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

    # Get input shape for in_s
    real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list)
    real_down = real_down[0]
    in_s = torch.zeros_like(real_down, device=opt.device)

    # Directory name
    s_dir_name = "logs_cmaes/%d" % (opt.experiment_id)

    # archive, emitter, and optimizer for cma-es
    n_features = 100 #number of input features for the noise vector generator

    #get the size of the noise map that TOAD-GAN will need
    vec_size = 0
    n_pad = int(1*opt.num_layer)
    for noise_map in noise_maps:
        nzx = int(round((noise_map.shape[-2] - n_pad * 2) * opt.scale_v))
        nzy = int(round((noise_map.shape[-1] - n_pad * 2) * opt.scale_h))
        vec_size += 12*nzx*nzy*opt.num_samples

    #create a random network that will take a vector from the emitter and generate a larger vector to feed into toad-gan
    rand_network = create_random_network(n_features, vec_size, opt.device).to(opt.device)
    state_dict = torch.load(s_dir_name + "/model")
    rand_network.load_state_dict(state_dict)
    rand_network.eval()

    #load the pickled archive
    df = pandas.read_pickle(s_dir_name + "/archive.zip")
 
    #set up a new archive
    n_bins = [20,20]
    archive_size = [(0, 200), (0, 100)]
    n_features = 100
    archive = GridArchive(n_bins, archive_size)
    archive.initialize(n_features)

    #populate the archive from the dataframe
    for _, row in df.iterrows():
        latent = np.array(row.loc["solution_0":])
        bcs = row.loc[["behavior_0", "behavior_1"]]
        obj = row.loc[["objective"]][0]
        archive.add(solution = latent, objective_value = obj, behavior_values = bcs)

    bc0 = float(input("Estimated value for behavior 0 (x axis): "))
    bc1 = float(input("Estimated value for behavior 1 (y axis): "))
    
    elite = archive.elite_with_behavior([bc0, bc1])

    if type(elite[0]) is not np.ndarray:
        print("This elite does not exist. Please try again.")
        exit()
    
    solution = torch.from_numpy(elite[0]).float()

    noise = rand_network(solution).detach()
    
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=1)

    level = levels[0]
    ascii_level = one_hot_to_ascii_level(level, opt.token_list)

    img = opt.ImgGen.render(ascii_level)
    img.save("%s/elite_%d_%d.png" % (s_dir_name, int(bc0), int(bc1)))