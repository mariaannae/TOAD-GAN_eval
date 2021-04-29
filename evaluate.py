import os
#from shutil import copyfile
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas

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

from metrics import platform_test_vec, num_jumps
from random_network import create_random_network
from playability import test_playability

from mario.level_image_gen import LevelImageGen
from utils import LevelObject
from py4j.java_gateway import JavaGateway
from tkinter import *
from PIL import ImageTk, Image

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

# Path to the AI Framework jar for Playing levels
MARIO_AI_PATH = os.path.abspath(os.path.join(os.path.curdir, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar"))

#define the fitness function:
def fit_func(solution, device, generators, num_layer, rand_network, reals, noise_amplitudes, opt, ImgGen, level_l, level_h,
             is_loaded, use_gen, error_msg, game, gateway, render_mario, in_s, scale_v, scale_h, save_dir, num_samples):

    #create the noise generator
    solution = torch.tensor(solution).float().to(device)
    noise_vector = rand_network(solution).flatten().to(device)

    #generate levels
    levels = generate_samples_cmaes(generators, noise_maps, reals, noise_amplitudes, noise_vector, opt, in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    score_play = 0.0
    score_platform = 0.0
    score_jumps = 0.0

    for level in levels:

        playable = test_playability(level, opt.token_list, ImgGen, level_l, level_h, is_loaded, use_gen, error_msg, game, gateway, render_mario)
        score_play+=playable

        score_platform += platform_test_vec(level, opt.token_list)

        score_jumps += num_jumps(level, opt.token_list)

    score_play = (score_play/float(len(levels)))
    score_platform = float(score_platform)/float(len(levels))
    score_jumps = float(score_jumps)/float(len(levels))

    return score_play, score_platform, score_jumps

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
    
    #setting number of samples generated to 1, so that each noise vector produced corresponds to only one generated level and is not divided into multiple levels
    opt.num_samples = 1

    render_mario = True
    root = Tk(className=" TOAD-GUI")

    level_l = IntVar()
    level_h = IntVar()
    level_l.set(0)
    level_h.set(0)
    placeholder = Image.new('RGB', (890, 256), (255, 255, 255))  # Placeholder image for the preview
    load_string_gen = StringVar()
    load_string_txt = StringVar()
    ImgGen = LevelImageGen(os.path.join(os.path.join(os.curdir, "utils"), "sprites"))
    use_gen = BooleanVar()
    use_gen.set(False)
    levelimage = ImageTk.PhotoImage(placeholder)
    level_obj = LevelObject('-', None, levelimage, ['-'], None, None)
    is_loaded = BooleanVar()
    is_loaded.set(False)
    error_msg = StringVar()
    error_msg.set("No Errors")

    # Path to the AI Framework jar for Playing levels
    MARIO_AI_PATH = os.path.abspath(os.path.join(os.path.curdir, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar"))

    # Py4j Java bridge uses Mario AI Framework
    gateway = JavaGateway.launch_gateway(classpath=MARIO_AI_PATH, die_on_exit=True, redirect_stdout=sys.stdout,
                                         redirect_stderr=sys.stderr)

    # Open up game window and assign agent
    game = gateway.jvm.engine.core.MarioGame()
    game.initVisuals(2.0)
    agent = gateway.jvm.agents.robinBaumgarten.Agent()
    game.setAgent(agent)


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


    #logging setup
    if not os.path.exists('logs_cmaes'):
        os.mkdir('logs_cmaes')

    i = 1
    while(os.path.exists('logs_cmaes/'+ str(i))):
        i+=1

    logdir = "logs_cmaes/"+str(i)
    os.mkdir(logdir)

##############################################################################################
#pyribs implementation

    # archive, emitter, and optimizer for cma-es
    n_features = 100 #number of input features for the noise vector generator
    batch_size = 10
    archive = GridArchive([20,20], [(0, 50), (0, 40)]) # objs are platform mismatches, jumps
    emitters = [ImprovementEmitter(
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
    #pyribs ask/tell loop
    n_iter =10000
    for i in range(n_iter):
        solutions = optimizer.ask()
        #solutions =(torch.from_numpy(solutions).float()).to(opt.device)

        bcs = []
        objectives = []
        playable = 0

        platform = 0
        num_levels = 0
        for solution in solutions:
            result = fit_func(solution, opt.device, generators, opt.num_layer, rand_network, reals, noise_amplitudes, opt,
                           ImgGen, level_l, level_h, is_loaded, use_gen, error_msg, game, gateway, render_mario,
                           in_s=in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name,
                           num_samples=opt.num_samples)    
            
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

        if i % 5 == 0:
            print("Saving after generation: ", i)
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

            #generate a playability plot
            num_levelgen = opt.num_samples * len(solutions)
            fig = plt.figure()
            xs = [i+1 for i, _ in enumerate(percent_playable)]
            plt.bar(xs, percent_playable)
            plt.xlabel('Generations')
            plt.ylabel('Playable')
            plt.title("Average percentage of level completed by A* agent at each generation")
            plt.savefig(logdir + '/playable')
            plt.close()

            #generate a platform score plot
            fig = plt.figure()
            xs = [i+1 for i, _ in enumerate(platform_score)]
            plt.bar(xs, platform_score)
            plt.xlabel('Generations')
            plt.ylabel('Platform solidity')
            plt.title("Average platform holes")
            plt.savefig(logdir + '/platform')
            plt.close()

            #save the archive
            df = archive.as_pandas(include_solutions=True)
            df.to_pickle(logdir + "/archive.zip")


        