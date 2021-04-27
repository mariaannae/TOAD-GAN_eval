import os
import torch

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level, place_a_mario_token

from utils import play_level, LevelObject
from PIL import ImageTk

def test_playability(vec, token_list, ImgGen, level_l, level_h, is_loaded, use_gen, error_msg, game, gateway, render_mario):

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
    perc = play_level(level_obj, game, gateway, render_mario)
    return perc



