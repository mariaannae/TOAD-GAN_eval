import os
import torch

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen


from utils import load_level, play_level
from py4j.java_gateway import JavaGateway
from tkinter import *

# Path to the AI Framework jar for Playing levels
MARIO_AI_PATH = os.path.abspath(os.path.join(os.path.curdir, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar"))

def test_playability(vec, token_list):

    # Py4j Java bridge uses Mario AI Framework
    gateway = JavaGateway.launch_gateway(classpath=MARIO_AI_PATH, die_on_exit=True, redirect_stdout=sys.stdout,
                                         redirect_stderr=sys.stderr)

    # Open up game window and assign agent
    game = gateway.jvm.engine.core.MarioGame()
    game.initVisuals(2.0)
    agent = gateway.jvm.agents.robinBaumgarten.Agent()
    game.setAgent(agent)

    #create a level object to load the level into
    level_obj = LevelObject()

    #convert the level to ascii
    level_obj.ascii_level = one_hot_to_ascii_level(vec.detach(), token_list)

    # Check if a Mario token exists - if not, we need to place one
    m_exists = False
    for line in level_obj.ascii_level:
        if 'M' in line:
            m_exists = True
            break

    if not m_exists:
        level_obj.ascii_level = place_a_mario_token(level_obj.ascii_level)
        level_obj.tokens = token_list

        img = ImageTk.PhotoImage(ImgGen.render(level_obj.ascii_level))
        level_obj.image = img

        level_obj.scales = None
        level_obj.noises = None

        level_l.set(lev.shape[-1])
        level_h.set(lev.shape[-2])

        is_loaded.set(True)
        use_gen.set(False)
        error_msg.set("Level loaded")

    # Play the level
    perc = play_level(level_obj, game, gateway, render_mario)
    print(perc)
    print()



