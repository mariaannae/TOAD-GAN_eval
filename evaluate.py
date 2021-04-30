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

from xvfbwrapper import Xvfb
import ray


#define the fitness function:
@ray.remote
def multi_fit_func(solution, device, generators, num_layer, rand_network, reals, noise_amplitudes, opt, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    solution = torch.tensor(solution).float().to(device)
    noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    score_play = 0.0
    score_platform = 0.0
    score_jumps = 0.0

    for level in levels:

        playable = test_playability(level, opt.token_list)
        score_play+=playable

        score_platform += platform_test_vec(level, opt.token_list)

        score_jumps += num_jumps(level, opt.token_list)

    score_play = (score_play/float(len(levels)))
    score_platform = float(score_platform)/float(len(levels))
    score_jumps = float(score_jumps)/float(len(levels))

    return score_play, score_platform, score_jumps

def  fit_func(solution, device, generators, num_layer, rand_network, reals, noise_amplitudes, opt, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    solution = torch.tensor(solution).float().to(device)
    noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    score_play = 0.0
    score_platform = 0.0
    score_jumps = 0.0

    for level in levels:

        playable = test_playability(level, opt.token_list)
        score_play+=playable

        score_platform += platform_test_vec(level, opt.token_list)

        score_jumps += num_jumps(level, opt.token_list)

    score_play = (score_play/float(len(levels)))
    score_platform = float(score_platform)/float(len(levels))
    score_jumps = float(score_jumps)/float(len(levels))

    return score_play, score_platform, score_jumps


def tb_logging(archive, itr, start_time, logdir, score):
    # TensorBoard Logging
    df = archive.as_pandas(include_solutions=False)
    elapsed_time = time.time() - start_time
    writer.add_scalar('score/mean', df['objective'].mean(), itr)
    writer.add_scalar('score/max', df['objective'].max(), itr)
    writer.add_scalar('score/min', df['objective'].min(), itr)
    writer.add_scalar('playability', score, itr)
    writer.add_scalar('generations/second', elapsed_time, itr)

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
    parse.add_argument("--token_insert_experiment", action="store_true", help="make token insert experiment (experimental!)", default=False)
    parse.add_argument("--multiproc", action="store_true", help="run with multiprocessing", default=False)

    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
            parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)
    
    #setting number of samples generated to 1, so that each noise vector produced corresponds to only one generated level and is not divided into multiple levels
    opt.num_samples = 1

    #TODO: maybe fix this option later in the project
    if opt.make_mario_samples:
        # Code to make a large body of mario samples for other experiments
        opt.game = 'mario'
        sprite_path = opt.game + '/sprites'
        opt.ImgGen = MarioLevelGen(sprite_path)
        opt.gen_start_scale = 0  # Forced for this experiment

        generate_mario_samples(opt)
       
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
        prefix = "arbitrary"

        # Directory name
        s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)


    #logging setup
    if not os.path.exists('logs_cmaes'):
        os.mkdir('logs_cmaes')

    i = 1
    while(os.path.exists('logs_cmaes/'+ str(i))):
        i+=1

    logdir = "logs_cmaes/"+str(i)
    os.mkdir(logdir)

    if not(os.path.exists('logs_cmaes/tb_logs/')):
        os.mkdir('logs_cmaes/tb_logs/')

    tb_logdir = "logs_cmaes/tb_logs/"+str(i)
    os.mkdir(tb_logdir)
    writer = SummaryWriter(tb_logdir)

##############################################################################################
#evolutionary search implementation

    # archive, emitter, and optimizer for cma-es
    n_features = 100 #number of input features for the noise vector generator
    batch_size = 10
    archive = GridArchive([20,20], [(0, 200), (0, 100)]) # objs are platform mismatches, jumps
    emitters = [OptimizingEmitter(
        archive,
        np.zeros(n_features),
        1.0,
        batch_size=batch_size,
    )]
    optimizer = Optimizer(archive, emitters)


    #get the size of the noise map that TOAD-GAN will need
    vec_size = 0
    n_pad = int(1*opt.num_layer)
    for noise_map in noise_maps:
        nzx = int(round((noise_map.shape[-2] - n_pad * 2) * opt.scale_v))
        nzy = int(round((noise_map.shape[-1] - n_pad * 2) * opt.scale_h))
        vec_size += 12*nzx*nzy*opt.num_samples

    #create a random network that will take a vector from the emitter and generate a larger vector to feed into toad-gan
    rand_network = create_random_network(n_features, vec_size, opt.device).to(opt.device)

    #the model will not be trained, so we only need to save it once for reproduceability
    torch.save(rand_network.state_dict(), logdir+"/model")

    percent_playable = []
    platform_score = []

    #create virtual display
    xvfb = Xvfb()
    xvfb.start()

    if opt.multiproc:
        ray.init()

    #pyribs ask/tell loop
    n_generation =10000
    
    for i in range(n_generation):
        
        start_time = time.time()
        solutions = optimizer.ask()
        solutions =(torch.from_numpy(solutions).float()).to(opt.device)

        bcs = []
        objectives = []
        playable = 0

        platform = 0
        num_levels = 0

        if opt.multiproc:
            futures = [multi_fit_func.remote(solution, opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]
            results = ray.get(futures)
        else:
            results = [fit_func(solution, opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]

        for result in results:
            bcs.append([result[1], result[2]])
            objectives.append(result[0])

            #playability tracking
            playable+=result[0]/opt.num_samples
            num_levels+=opt.num_samples

            #platform tracking
            platform+=result[1]

        playable = (float(playable)/float(num_levels))
        percent_playable.append(playable)

        platform = float(platform)/float(num_levels)
        platform_score.append(platform)

        optimizer.tell(objectives, bcs)

        tb_logging(archive, i, start_time, logdir, playable)

        if i % 10 == 0:
            #generate a heatmap
            plt.figure(figsize=(8,6))
            grid_archive_heatmap(archive)
            plt.title("Playability")
            plt.xlabel("Platform Solidity")
            plt.ylabel("Number of Jumps")
            figname = '/map_' + str(i)
            #figname = 'test'
            plt.savefig(logdir + figname)
            #plt.show()
            plt.close()

            #save the archive
            df = archive.as_pandas(include_solutions=True)
            df.to_pickle(logdir + "/archive.zip")


        