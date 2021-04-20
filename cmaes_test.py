
import os
#from shutil import copyfile
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from config import get_arguments, post_config
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from generate_noise import generate_spatial_noise
from generate_samples_cmaes import generate_samples_cmaes
from models import load_trained_pyramid

import numpy as np
import torch

from cma.evolution_strategy import CMAEvolutionStrategy
import cma

from evaluate import platform_test_vec
from random_network import create_random_network

#define the fitness function:
def fit_func(solution, device, generators, num_layer, vec_size, reals, noise_amplitudes, opt, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    rand_network = create_random_network(len(solution), vec_size, device).to(device)
    solution = torch.tensor(solution).float().to(device)
    noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)
        
    #evaluate levels (using placeholder metric of platform solidity)
    for level in levels:
        score = 0
        for level in levels:
            score += platform_test_vec(level)
    score = float(score)/float(len(levels))

    return score

if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    # Parse arguments
    parse = get_arguments()
    parse.add_argument("--out_", help="folder containing generator files", default="output/wandb/latest-run/files/")
    parse.add_argument("--scale_v", type=float, help="vertical scale factor", default=1.0)
    parse.add_argument("--scale_h", type=float, help="horizontal scale factor", default=1.0)
    parse.add_argument("--gen_start_scale", type=int, help="scale to start generating in", default=0)
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=10)
    parse.add_argument("--make_mario_samples", action="store_true", help="make 1000 samples for each mario generator"
                                                                         "specified in the code.", default=False)
    parse.add_argument("--seed_mariokart_road", action="store_true", help="seed mariokart generators with a road image", default=False)
    parse.add_argument("--token_insert_experiment", action="store_true", help="make token insert experiment (experimental!)", default=False)
    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
            parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)
    

    #TODO: maybe fix this option later in the project
    if opt.make_mario_samples:
        # Code to make a large body of mario samples for other experiments
        opt.game = 'mario'
        sprite_path = opt.game + '/sprites'
        opt.ImgGen = MarioLevelGen(sprite_path)
        opt.gen_start_scale = 0  # Forced for this experiment

        generate_mario_samples(opt)

    #TODO: maybe fix this option later in the project
    elif opt.seed_mariokart_road:
        # Code to make mario kart seeded road examples
        opt.game = 'mariokart'
        sprite_path = opt.game + '/sprites'
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling
        opt.gen_start_scale = 0  # Forced for this experiment

        # Load level
        real = read_level(opt, None, replace_tokens).to(opt.device)
        # Load generator
        generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

        # Define paths to seed road image(s)
        seed_road_images = ['mariokart/seed_road/seed_road.png']
        # seed_road_images = ['mariokart/seed_road/MNIST_examples/eights/sample_%d.png' % im for im in range(20)]

        for i, img_path in enumerate(seed_road_images):
            # Read and convert seed road image
            seed_road_img = plt.imread(img_path)
            opt.seed_road = torch.Tensor(1 - seed_road_img[:, :, 0])

            # Scales have to be fitting with seed road image (preferably make seed road the size of scale 0 directly!)
            scale_0_h = reals[0].shape[-1] / reals[-1].shape[-1]
            scale_0_v = reals[0].shape[-2] / reals[-1].shape[-2]
            shape_r_h = round(opt.seed_road.shape[-1] / scale_0_h)
            shape_r_v = round(opt.seed_road.shape[-2] / scale_0_v)
            scale_h = shape_r_h / reals[-1].shape[-1]
            scale_v = shape_r_v / reals[-1].shape[-2]

            real_down = downsample(1, [[scale_v, scale_h]], real, opt.token_list)
            real_down = real_down[0]

            # in_s = torch.zeros((round(reals[-1].shape[-2]*scale_v), round(reals[-1].shape[-1]*scale_h)),
            in_s = torch.zeros(real_down.shape,
                                device=opt.device)  # necessary for correct input shape

            # Directory name
            s_dir_name = "random_road_samples_v%.5f_h%.5f_st%d_%d" % (opt.scale_v, opt.scale_h, opt.gen_start_scale, i)

            # Generate samples
            generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                            scale_v=scale_v, scale_h=scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    else:
        # Code to make samples for given generator
        token_insertion = True if opt.token_insert and opt.token_insert_experiment else False

        # Init game specific inputs
        replace_tokens = {}
        sprite_path = opt.game + '/sprites'
        if opt.game == 'mario':
            opt.ImgGen = MarioLevelGen(sprite_path)
            replace_tokens = MARIO_REPLACE_TOKENS
            downsample = special_mario_downsampling

        elif opt.game == 'mariokart':
            opt.ImgGen = MariokartLevelGen(sprite_path)
            replace_tokens = MARIOKART_REPLACE_TOKENS
            downsample = special_mariokart_downsampling

        else:
            NameError("name of --game not recognized. Supported: mario, mariokart")

        # Load level
        real = read_level(opt, None, replace_tokens).to(opt.device)
        # Load Generator
        generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

        # Get input shape for in_s
        real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list)
        real_down = real_down[0]
        in_s = torch.zeros_like(real_down, device=opt.device)
        prefix = "arbitrary"

        # Directory name
        s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)



##############################################################################################
#cma-es implementation

    cma_opts = cma.CMAOptions()
    cma_opts.set('timeout', 5*60) #set timeout in seconds
    cma_opts.set('tolflatfitness', 10)
    n_features = 100 #number of input features for the noise vector generator. other tolerance options available.

    #get the size of the noise map that TOAD-GAN will need
    vec_size = 0
    n_pad = int(1*opt.num_layer)
    for noise_map in noise_maps:
        nzx = int(round((noise_map.shape[-2] - n_pad * 2) * opt.scale_v))
        nzy = int(round((noise_map.shape[-1] - n_pad * 2) * opt.scale_h))
        vec_size += 12*nzx*nzy*opt.num_samples


    #initialize cma-es
    #TODO:revisit sigma
    es = CMAEvolutionStrategy([0]*n_features, sigma0 = 1)

    cma_opts.set('timeout', 5*60) #set timeout in seconds
    while not es.stop():
        solutions = es.ask()

        #calculate fitness
        objectives = []
        for solution in solutions:
            obj = fit_func(solution, opt.device, generators, opt.num_layer, vec_size, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)
            objectives.append(obj)

        es.tell(solutions, objectives)
        es.logger.add()
        es.disp()
   
