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

from metrics import platform_test_vec, num_jumps, hamming_dist, compute_kl_divergence, normalized_compression_dist, num_enemies, num_koopa, midair_pipes, new_tile_types, enemy_on_stairs, spiky
from random_network import create_random_network
from playability import test_playability

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from torchvision import transforms

from cma import CMAEvolutionStrategy

import ray


#define the fitness function:
@ray.remote
def multi_fit_func(solution, device, generators, num_layer, rand_network, reals, noise_amplitudes, opt, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    solution = solution.clone().detach()
    with torch.no_grad():
        noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    score_play = 0.0
    score_unplay = 0.0
    score_platform = 0.0
    score_hamming = 0.0
    score_kl_divergence = 0.0
    score_time = 0.0
    score_max_jump = 0.0
    score_jumps = 0.0
    score_ncd = 0.0
    mario_end_state = -1 #0 if completed, 1 if timeout, 2 if killed
    score_enemies = 0.0
    score_koopas = 0.0
    score_pipes = 0.0
    score_new_tiles = 0.0
    score_spiky = 0.0
    score_enemy_on_stairs = 0.0
    score_pipes_and_platform = 0.0


    for level in levels:

        perc, timeLeft, jumps, max_jump = test_playability(level, opt.token_list)
        
        if perc == 100:
            mario_end_state = 0
        else:
            if timeLeft == 0:
                mario_end_state = 1
            else:
                timeLeft = 0
                mario_end_state = 2
            
        score_play += perc
        score_unplay += 100-perc
        score_jumps += jumps
        score_max_jump += max_jump
        score_time += timeLeft
        
        score_new_tiles += new_tile_types(level, opt)
        score_pipes -= midair_pipes(level, opt)
        score_koopas += num_koopa(level, opt)
        score_enemies += num_koopa(level, opt)
        score_ncd += normalized_compression_dist(level, opt)
        score_platform -= platform_test_vec(level, opt.token_list)
        score_hamming += hamming_dist(level, opt)
        score_spiky = spiky(level, opt)
        score_enemy_on_stairs += enemy_on_stairs(level, opt)

        kl, _ = compute_kl_divergence(level, opt)
        score_kl_divergence += kl

    score_pipes_and_platform = score_pipes + score_platform
    
    obj_dict = {"pipes_and_platform": score_pipes_and_platform, "enemy_on_stairs": score_enemy_on_stairs, "spiky": score_spiky, "new_tiles": score_new_tiles, "pipes": score_pipes, "koopas": score_koopas, "enemies": score_enemies, "ncd": score_ncd, "hamming": score_hamming, "playable": score_play, "tpkl": score_kl_divergence, "unplayable": score_unplay, "platform": score_platform}
    bc_dict = {"enemy_on_stairs": score_enemy_on_stairs, "spiky": score_spiky, "new_tiles": score_new_tiles, "pipes": score_pipes, "koopas": score_koopas, "enemies": score_enemies, "ncd": score_ncd, "hamming": score_hamming, "playable": score_play, "tpkl": score_kl_divergence, "time": score_time, "max_jump": score_max_jump, "n_jumps": score_jumps, "platform": score_platform}

    return obj_dict[opt.obj], bc_dict[opt.bcs[0]], bc_dict[opt.bcs[1]], mario_end_state
    

def fit_func(solution, device, generators, num_layer, rand_network, reals, noise_amplitudes, opt, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    solution = solution.clone().detach()
    with torch.no_grad():
        noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    score_play = 0.0
    score_unplay = 0.0
    score_platform = 0.0
    score_hamming = 0.0
    score_kl_divergence = 0.0
    score_time = 0.0
    score_max_jump = 0.0
    score_jumps = 0.0
    score_ncd = 0.0
    mario_end_state = -1 #0 if completed, 1 if timeout, 2 if killed
    score_enemies = 0.0
    score_koopas = 0.0
    score_pipes = 0.0
    score_new_tiles = 0.0
    score_spiky = 0.0
    score_enemy_on_stairs = 0.0
    score_pipes_and_platform = 0.0


    for level in levels:

        perc, timeLeft, jumps, max_jump = test_playability(level, opt.token_list)
        
        if perc == 100:
            mario_end_state = 0
        else:
            if timeLeft == 0:
                mario_end_state = 1
            else:
                timeLeft = 0
                mario_end_state = 2
            
        score_play += perc
        score_unplay += 100-perc
        score_jumps += jumps
        score_max_jump += max_jump
        score_time += timeLeft
        
        score_new_tiles += new_tile_types(level, opt)
        score_pipes -= midair_pipes(level, opt)
        score_koopas += num_koopa(level, opt)
        score_enemies += num_koopa(level, opt)
        score_ncd += normalized_compression_dist(level, opt)
        score_platform -= platform_test_vec(level, opt.token_list)
        score_hamming += hamming_dist(level, opt)
        score_spiky = spiky(level, opt)
        score_enemy_on_stairs += enemy_on_stairs(level, opt)

        kl, _ = compute_kl_divergence(level, opt)
        score_kl_divergence += kl

    score_pipes_and_platform = score_pipes + score_platform
    
    obj_dict = {"pipes_and_platform": score_pipes_and_platform, "enemy_on_stairs": score_enemy_on_stairs, "spiky": score_spiky, "new_tiles": score_new_tiles, "pipes": score_pipes, "koopas": score_koopas, "enemies": score_enemies, "ncd": score_ncd, "hamming": score_hamming, "playable": score_play, "tpkl": score_kl_divergence, "unplayable": score_unplay, "platform": score_platform}
    bc_dict = {"enemy_on_stairs": score_enemy_on_stairs, "spiky": score_spiky, "new_tiles": score_new_tiles, "pipes": score_pipes, "koopas": score_koopas, "enemies": score_enemies, "ncd": score_ncd, "hamming": score_hamming, "playable": score_play, "tpkl": score_kl_divergence, "time": score_time, "max_jump": score_max_jump, "n_jumps": score_jumps, "platform": score_platform}

    return obj_dict[opt.obj], bc_dict[opt.bcs[0]], bc_dict[opt.bcs[1]], mario_end_state
 
def tb_logging(archive, itr, start_time, logdir, score, bc0, bc1, end_states):
    # TensorBoard Logging
    
    if type(archive) is not int: 
        df = archive.as_pandas(include_solutions=False)
        writer.add_scalar('score/mean', df['objective'].mean(), itr)
        writer.add_scalar('score/max', df['objective'].max(), itr)
        writer.add_scalar('score/min', df['objective'].min(), itr)

    completed = [1 for entry in end_states if entry==0]
    timeouts = [1 for entry in end_states if entry==1]
    killed = [1 for entry in end_states if entry==2]

    elapsed_time = time.time() - start_time
    writer.add_scalar('objective', score, itr)
    writer.add_scalar('behavior 0', bc0, itr)
    writer.add_scalar('behavior 1', bc1, itr)
    writer.add_scalar("completed", 100*float(sum(completed))/float(len(end_states)), itr)
    writer.add_scalar("timeouts", 100*float(sum(timeouts))/float(len(end_states)), itr)
    writer.add_scalar("killed", 100*float(sum(killed))/float(len(end_states)), itr)
    writer.add_scalar('seconds/generation', elapsed_time, itr)



if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    # Parse arguments
    parse = get_arguments()
    parse.add_argument("--out_", help="folder containing generator files", default="output/wandb/latest-run/files/")
    parse.add_argument("--scale_v", type=float, help="vertical scale factor", default=1.0)
    parse.add_argument("--scale_h", type=float, help="horizontal scale factor", default=1.0)
    parse.add_argument("--gen_start_scale", type=int, help="scale to start generating in", default=0)
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=1)
    parse.add_argument("--multiproc", action="store_true", help="run with multiprocessing", default=False)
    parse.add_argument("--vdisplay", action="store_true", help="run with a virtual display", default=False)
    parse.add_argument("--pycma", action="store_true", help="run with pycma instead of pyribs", default = False)
    parse.add_argument("--obj", type=str, help = "fitness measure", default = "playable", choices=["pipes_and_platform", "spiky", "enemy_on_stairs", "new_tiles", "pipes", "koopas", "enemies", "ncd", "hamming", "playable", "tpkl", "unplayable", "platform"])
    parse.add_argument("--bcs", type=str, help = "fitness measure", nargs = 2, required = True, choices=["spiky", "enemy_on_stairs", "new_tiles", "pipes", "koopas", "enemies", "ncd", "hamming", "playable", "tpkl", "time", "max_jump", "n_jumps", "n_enemies", "platform"])
    parse.add_argument("--cma_me", action="store_true", help = "run with an improvement emitter rather than an optimizing emitter", default=False)
    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
            parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)
    bc_names = {"spiky": "Number of Spiky Enemies", "enemy_on_stairs": "Enemies Placed on Pyramid Blocks", "new_tiles": "Tile Types not in Reference Level", "pipes": "Midair Pipes", "koopas": "Number of Green Koopas", "enemies": "Number of Enemies", "ncd": "Normalized Compression Distance", "platform": "Platform Holes( by Tile)", "hamming": "Hamming Distance", "playable": "Percentage of Level Completed", "tpkl": "Tile Pattern KL Divergence", "time": "Time Remaining After Play", "max_jump": "Maximum Jump Width", "n_jumps": "Number of Jumps Taken", "n_enemies": "Number of Enemies", "platform": "Platform Holes(by tile)"}
    bc_ranges = {"spiky": (0, 10), "enemy_on_stairs": (0, 30), "new_tiles": (0, 50), "pipes": (-20, 0), "koopas": (0, 10), "enemies": (0, 100), "ncd": (0, 1), "hamming": (0, 1), "playable": (0, 100), "tpkl": (0, 10), "time": (0, int(1.2e4)), "max_jump": (0, 200), "n_jumps": (0, 100), "n_enemies": (0, 50), "platform": (0, 203)}
    obj_names = {"pipes_and_platform": "Platform Holes and Pipes in Midair", "spiky": "Number of Spiky Enemies", "enemy_on_stairs": "Enemies Placed on Pyramid Blocks", "new_tiles": "Tile Types not in Reference Level", "pipes": "Midair Pipes", "koopas": "Number of Green Koopas", "enemies": "Number of Enemies", "ncd": "Normalized Compression Distance", "hamming": "Hamming Distance", "playable": "Percentage of Level Completed", "tpkl": "Tile Pattern KL Divergence", "unplayable": "Percentage of Level Left Incomplete After Play", "max_jump": "Maximum Jump Width", "n_jumps": "Number of Jumps Taken", "n_enemies": "Number of Enemies", "platform": "Platform Holes(by tile)"}

    if opt.bcs[0][1] == 1:
        bins0 = 50
    elif opt.bcs[0] == "tpkl":
        bins0 = 100
    else:
        bins0 = min(bc_ranges[opt.bcs[0]][1]-bc_ranges[opt.bcs[0]][0], 100)

    if opt.bcs[1][1] == 1:
        bins1 = 50
    elif opt.bcs[1] == "tpkl":
        bins1 = 100
    else:
        bins1 = min(bc_ranges[opt.bcs[1]][1]-bc_ranges[opt.bcs[1]][0], 100)



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
#pyribs implementation
    if not opt.pycma:
        # archive, emitter, and optimizer for cma-es
        n_features = 100 #number of input features for the noise vector generator
        batch_size = 10
        n_bins = [bins0, bins1]
        archive = GridArchive(n_bins, [bc_ranges[opt.bcs[0]], bc_ranges[opt.bcs[1]]]) # behavior 0, behavior 1
        if opt.cma_me:
            emitters = [ImprovementEmitter(
                archive,
                np.zeros(n_features),
                1.0,
                batch_size=batch_size
                ) for _ in range(1)]
        else:
            emitters = [OptimizingEmitter(
            archive,
            np.zeros(n_features),
            1.0,
            batch_size=batch_size
            ) for _ in range(1)]

            for key in obj_names:
                obj_names[key] = "Optimizing for "+obj_names[key]

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
        rand_network.eval()

        #the model will not be trained, so we only need to save it once for reproduceability
        torch.save(rand_network.state_dict(),  logdir+"/model")

        #create virtual display
        if opt.vdisplay:
            from xvfbwrapper import Xvfb
            xvfb = Xvfb()
            xvfb.start()

        if opt.multiproc:
            ray.init()

        #pyribs ask/tell loop
        n_generation =100000
    
        for i in range(n_generation):
            
            start_time = time.time()
            solutions = optimizer.ask()
            solutions =(torch.from_numpy(solutions).float()).to(opt.device)

            bcs = []
            objectives = []
            
            if opt.multiproc:
                futures = [multi_fit_func.remote(solution, opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]
                results = ray.get(futures)
            else:
                results = [fit_func(solution, opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]

            
            bc0 = 0
            bc1 = 0
            end_states = []

            for result in results:
                bcs.append([result[1], result[2]])
                objectives.append(result[0])
                bc0+=result[1]
                bc1+=result[2]
                end_states.append(result[3])

            optimizer.tell(objectives, bcs)

            score_obj = max(objectives)
            bc0 = bc0/float(len(objectives))
            bc1 = bc1/float(len(objectives))

            tb_logging(archive, i, start_time, logdir, score_obj, bc0, bc1, end_states)

            if i % 100 == 0:
                #generate a heatmap
                plt.figure(figsize=(8,6))
                #plt.figure(figsize=(8,2))
                grid_archive_heatmap(archive)
                plt.title(obj_names[opt.obj])
                plt.xlabel(bc_names[opt.bcs[0]]) #objective 0
                plt.ylabel(bc_names[opt.bcs[1]]) #objective 1

                '''
                plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left=False,      # ticks along the bottom edge are off
                    right=False,         # ticks along the top edge are off
                    labelleftFalse) # labels along the bottom edge are off
                '''

                figname = '/map_' + str(i)
                plt.savefig(logdir + figname)
                plt.close()

                #save the archive
                df = archive.as_pandas(include_solutions=True)
                df.to_pickle(logdir + "/archive.zip")


    else:
    ##############################################################################################
    #cma/pycma implementation
    
        n_features = 100 #number of input features for the noise vector generator. other tolerance options available.

        #get the size of the noise map that TOAD-GAN will need
        vec_size = 0
        n_pad = int(1*opt.num_layer)
        for noise_map in noise_maps:
            nzx = int(round((noise_map.shape[-2] - n_pad * 2) * opt.scale_v))
            nzy = int(round((noise_map.shape[-1] - n_pad * 2) * opt.scale_h))
            vec_size += 12*nzx*nzy*opt.num_samples

        rand_network = create_random_network(n_features, vec_size, opt.device).to(opt.device)
        rand_network.eval()

        #the model will not be trained, so we only need to save it once for reproduceability
        torch.save(rand_network.state_dict(),  logdir+"/model")

        #create virtual display
        if opt.vdisplay:
            from xvfbwrapper import Xvfb
            xvfb = Xvfb()
            xvfb.start()

        if opt.multiproc:
            ray.init()

        #TODO:revisit sigma
        es = CMAEvolutionStrategy(torch.zeros(n_features), sigma0 = 5.0)

        i = 0
        n_generation = 10000
        while not es.stop() and i < n_generation:
            start_time = time.time()
            solutions = es.ask()

            #calculate fitness
            objectives = []
            bcs = []
            
            if opt.multiproc:
                futures = [multi_fit_func.remote(torch.from_numpy(solution).float(), opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]
                results = ray.get(futures)
            else:
                results = [fit_func(torch.from_numpy(solution).float(), opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples) for solution in solutions]

            playable = 0
            bc0 = 0
            bc1 = 0
            end_states = []

            for result in results:
                bcs.append([result[1], result[2]])
                objectives.append(result[0])
                playable+=result[0]
                bc0+=result[1]
                bc1+=result[2]
                end_states.append(result[3])

            es.tell(solutions, objectives)

            playable = 100 - playable/float(len(objectives))
            bc0 = bc0/float(len(objectives))
            bc1 = bc1/float(len(objectives))
            
            archive = 0
            tb_logging(archive, i, start_time, logdir, playable, bc0, bc1, end_states)

            
            i += 1

            