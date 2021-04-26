import random
import os

import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageDraw
from py4j.java_gateway import JavaGateway
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


def load_level(fname):
    level_l = IntVar()
    level_h = IntVar()
    level_l.set(0)
    level_h.set(0)
    placeholder = Image.new('RGB', (890, 256), (255, 255, 255))  # Placeholder image for the preview
    load_string_gen = StringVar()
    load_string_txt = StringVar()
    ImgGen = LevelImageGen(os.path.join(os.path.join(os.curdir, "utils"), "sprites"))
    error_msg = StringVar()
    error_msg.set("No Errors")
    use_gen = BooleanVar()
    use_gen.set(False)
    levelimage = ImageTk.PhotoImage(placeholder)
    level_obj = LevelObject('-', None, levelimage, ['-'], None, None)
    is_loaded = BooleanVar()
    is_loaded.set(False)

    # fname = fd.askopenfilename(title='Load Level', initialdir=os.curdir, filetypes=[("level .txt files", "*.txt")])
    if len(fname) == 0:
        return  # loading was cancelled
    try:
        error_msg.set("Loading level...")
        is_loaded.set(False)
        use_gen.set(False)

        if fname[-3:] == "txt":
            load_string_gen.set('Path: ' + fname)
            folder, lname = os.path.split(fname)

            # Load level
            lev, tok = read_level_from_file(folder, lname)

            level_obj.oh_level = torch.Tensor(lev)  # casting to Tensor to keep consistency with generated levels
            level_obj.ascii_level = one_hot_to_ascii_level(lev, tok)

            # Check if a Mario token exists - if not, we need to place one
            m_exists = False
            for line in level_obj.ascii_level:
                if 'M' in line:
                    m_exists = True
                    break

            if not m_exists:
                level_obj.ascii_level = place_a_mario_token(level_obj.ascii_level)
            level_obj.tokens = tok

            img = ImageTk.PhotoImage(ImgGen.render(level_obj.ascii_level))
            level_obj.image = img

            level_obj.scales = None
            level_obj.noises = None

            level_l.set(lev.shape[-1])
            level_h.set(lev.shape[-2])

            is_loaded.set(True)
            use_gen.set(False)
            error_msg.set("Level loaded")
        else:
            error_msg.set("No level file selected.")
    except Exception:
        error_msg.set("No level file selected.")
    return level_obj


def play_level(level_obj, game, gateway, render_mario):
    error_msg = StringVar()
    error_msg.set("No Errors")
    use_gen = BooleanVar()
    use_gen.set(False)
    is_loaded = BooleanVar()
    is_loaded.set(False)
    # error_msg.set("Playing level...")
    # is_loaded.set(False)
    # remember_use_gen = use_gen.get()
    # use_gen.set(False)
    # Py4j Java bridge uses Mario AI Framework
    # gateway = JavaGateway.launch_gateway(classpath=MARIO_AI_PATH, die_on_exit=True, redirect_stdout=sys.stdout, redirect_stderr=sys.stderr)
    # game = gateway.jvm.engine.core.MarioGame()
    perc = 0
    try:
        # game.initVisuals(2.0)
        # agent = gateway.jvm.agents.robinBaumgarten.Agent()
        # game.setAgent(agent)
        # while True:
        result = game.gameLoop(''.join(level_obj.ascii_level), 20, 0, render_mario, 1000000)
        perc = int(result.getCompletionPercentage() * 100)
        error_msg.set("Level Played. Completion Percentage: %d%%" % perc)
    except Exception:
        error_msg.set("Level Play was interrupted.")
        is_loaded.set(True)
        # use_gen.set(remember_use_gen)
    finally:
        # game.getWindow().dispose()
        gateway.java_process.kill()
        gateway.close()

    is_loaded.set(True)
    # use_gen.set(remember_use_gen)  # only set use_gen to True if it was previously
    return perc

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


def prepare_mnist_seed_images():
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('output/mnist/', train=False, download=True,
                                   transform=torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=1, shuffle=True)
    eights = torch.zeros((20, 1, 28, 28))
    e = 0
    while e < eights.shape[0]:
        batch = next(iter(test_loader))
        if batch[1].item() == 8:
            eights[e] = batch[0]
            e += 1

    for i in range(len(eights)):
        tmp = eights[i, 0]
        x, y = torch.where(tmp > 0)
        l_x = max(x) - min(x)
        l_y = max(y) - min(y)
        if l_x == l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            y_1 = min(y)
            y_2 = max(y) + 2
        elif l_x > l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            diff = l_x - l_y
            y_1 = min(y) - diff//2
            y_2 = max(y) + diff//2 + 2
        else:  # l_y > l_x:
            y_1 = min(y)
            y_2 = max(y) + 2
            diff = l_y - l_x
            x_1 = min(x) - diff//2
            x_2 = max(x) + diff//2 + 2
        tmp = tmp[x_1:x_2, y_1:y_2]
        # tmp = interpolate(tmp.unsqueeze(0).unsqueeze(0), (28, 28))
        plt.imsave('mariokart/seed_road/MNIST_examples/eights/sample_%d.png' % i, tmp[0][0], cmap='Greys')
