import os
import torch

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level, place_a_mario_token
from mario.level_image_gen import LevelImageGen

from utils import LevelObject
from py4j.java_gateway import JavaGateway
from tkinter import *
from PIL import ImageTk, Image, ImageDraw

# Path to the AI Framework jar for Playing levels
MARIO_AI_PATH = os.path.abspath(os.path.join(os.path.curdir, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar"))


def test_playability(vec, token_list):
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

    # Py4j Java bridge uses Mario AI Framework
    gateway = JavaGateway.launch_gateway(classpath=MARIO_AI_PATH, die_on_exit=True, redirect_stdout=sys.stdout,
                                         redirect_stderr=sys.stderr)

    # Open up game window and assign agent
    game = gateway.jvm.engine.core.MarioGame()
    game.initVisuals(2.0)
    agent = gateway.jvm.agents.robinBaumgarten.Agent()
    game.setAgent(agent)

    #create a level object to load the level into
    level_obj = LevelObject(0,0,0,0,0,0)

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

        #ImgGen = MarioLevelGen('utils/sprites')
        img = ImageTk.PhotoImage(ImgGen.render(level_obj.ascii_level))
        level_obj.image = img

        level_obj.scales = None
        level_obj.noises = None

        level_l.set(vec.shape[-1])
        level_h.set(vec.shape[-2])

        is_loaded.set(True)
        use_gen.set(False)
        error_msg.set("Level loaded")

    # Play the level
    #   
    perc = 0
    try:
        result = game.gameLoop(''.join(level_obj.ascii_level), 20, 0, render_mario, 1000000)
        perc = int(result.getCompletionPercentage() * 100)
        error_msg.set("Level Played. Completion Percentage: %d%%" % perc)
    except Exception:
        error_msg.set("Level Play was interrupted.")
        is_loaded.set(True)
    finally:
        # game.getWindow().dispose()
        gateway.java_process.kill()
        #gateway.close()
        gateway.shutdown()

    is_loaded.set(True)
    # use_gen.set(remember_use_gen)  # only set use_gen to True if it was previously
    return perc
    
    
    return perc



