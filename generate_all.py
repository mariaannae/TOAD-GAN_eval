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
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=1)
    parse.add_argument("--experiment_id", type=int, help="the experiment number to load from", default = 1)
    parse.add_argument("--n_features", type=int, help="the number of features passed to the random network by the emitter", default = 100)
    parse.add_argument("--all", action="store_true", help="generate all levels from the selected experiment", default=True)

    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
            parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)
    
    #setting number of samples generated to 1, so that each noise vector produced corresponds to only one generated level and is not divided into multiple levels
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
    n_features = opt.n_features #number of input features for the noise vector generator

    #get the size of the noise map that TOAD-GAN will need
    vec_size = 0
    print(opt.num_samples)
    n_pad = int(1*opt.num_layer)
    for noise_map in noise_maps:
        nzx = int(round((noise_map.shape[-2] - n_pad * 2) * opt.scale_v))
        nzy = int(round((noise_map.shape[-1] - n_pad * 2) * opt.scale_h))
        vec_size += 12*nzx*nzy*opt.num_samples

    for dirname in os.listdir("logs_cmaes"):
        s_dir_name = "logs_cmaes/"+dirname

        #create a random network that will take a vector from the emitter and generate a larger vector to feed into toad-gan
        rand_network = create_random_network(n_features, vec_size, opt.device).to(opt.device)
        try:
            state_dict = torch.load(s_dir_name + "/model")
       
            rand_network.load_state_dict(state_dict)
            rand_network.eval()

            #load the pickled archive
            df = pandas.read_pickle(s_dir_name + "/archive.zip")
        
            if opt.all:

                solutions = []
                bcs = []
                objs = []

                for _, row in df.iterrows():
                    latent = np.array(row.loc["solution_0":])
                    bc = row.loc[["behavior_0", "behavior_1"]]
                    obj = row.loc[["objective"]][0]

                    solutions.append(latent)
                    bcs.append(bc)
                    objs.append(obj)
                    
                for solution, bc, obj in zip(solutions, bcs, objs):
                    solution = torch.from_numpy(solution).float()
                    noise = rand_network(solution).detach()
                    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=1)
                    level = levels[0]
                    ascii_level = one_hot_to_ascii_level(level, opt.token_list)
                    img = opt.ImgGen.render(ascii_level)
                    img.save("%s/elite_%.3f_%.3f_score_%.2f.png" % (s_dir_name, bc[0], bc[1], obj))

        except:
            continue