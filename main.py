#!/usr/bin/env python
# coding: utf-8

# # Schneider Electric European Hackathon
# ## Challenge Data-Science: Zero Deforestation Mission
# - - -
# <img src="https://user-images.githubusercontent.com/116558787/201539352-ad0e29cb-501d-4644-9134-33a8b07ad693.PNG" alt="Schneider" width="800"/>
# 
# 
# ### Group SDG-UR: 
# - - -
# 
# - Ramón Sieira Martínez (SDG Group, Spain)
# - Francisco Javier Martínez de Pisón (University of La Rioja, Spain)
# - Jose Divasón (University of La Rioja, Spain)
# 
# ---
# 
# This Jupyter notebook allows one to train and create submissions for the Data-Schneider-European-Hackathon "Zero Deforestation Mission". It also contains the methodology and the general ideas that we have followed.
# 
# ##### Key ideas of our approach:
# - - -
# 
# 1. Training is performed with a 3-fold cross-validation. The model that obtains the best F1-Macro is saved.
# 2. The training dataset has been balanced.
# 3. The final prediction in the submission corresponds to an ensemble of 8 models, each one trained using a 5-fold cross-validation.
# 
# 
# ##### Detailed methodology:
# - - -
# 
# ###### Step 1: Tuning the learning rate, bath size and epochs
# 
# After a first visual inspection of the images contained in the dataset, we have performed several test in order to find an optimal batch size, also adjusting the learning rate and finding an adequate value of epochs needed.
# 
# We performed some test with a baseline approach. The goal at this step is to have some idea about the performance of a simple model.
# 
# We have also tried with different batch sizes (from 16 to 72), which is very important for us because, in principle, a larger batch size should speed up the calculations, which means we can perform more tests in less time (which is the key in a competition like this). Surprisingly, we have obtained slower and worse results with large bath sizes, even though we also tried with different learning rates (keeping it constant, increasing it linearly with respect to the learning rate, etc.). This has been very surprising and an unexpected result, but due to lack of time we have not studied this problem further. Therefore, we have proceeded with 16 as batch size, 0.00001 as learning rate and 100 epochs.
# 
# ###### Step 2: Finding augmentation filters
# 
# Data augmentation is a fundamental task when training any deep learning model. To try to make the process as fast as possible, we use the Kornia library to perform data augmentation on GPU instead of using CPU.
# 
# The first thing we have done is to test various types of data augmentation that are available in the library (separately). For this task, we used a model "tf_efficientnet_b3_ns".
# 
# We have tried the following augmentations. Due to the lack of time we have not performed many tests, only a few of them:
# 
# - HORIZONTAL_FLIP
# - VERTICAL_FLIP
# - ERASING_MAX_RATIO
# - COLORJITTER_BRIGHTNESS
# - COLORJITTER_CONTRAST
# - COLORJITTER_SATURATION
# - COLORJITTER_HUE
# - ROTATION_DEGREES
# - SHARPNESS_VALUE
# - MOTIONBLUR_KERNEL_SIZE
# - MOTIONBLUR_KERNEL_ANGLE
# - MOTIONBLUR_KERNEL_DIRECTION
# 
# The goal has been to find which augmentations really increase the final score. Some of them decrease the score, so we directly discarded them. We have discovered that some of them do produce a high increase, like saturation, sharpness, rotations and blur. It was not clear if some of the other filters really produce an improvement, so we did not perform further experiments with them to save up time.
# 
# ###### Step 3: Finding an augmentation range
# 
# Those filters with the best results are selected and their maximum and minimum values are determined. We have not performed many test to find an optimal range because here is not enough time for that, so sometimes we have had to estimate it based only on one or two results.
# 
# Finally, the following augmentations and ranges have been selected:
# ```
#     saturation_min_max = (0.01, 0.20) 
#     rotation_min_max = (0.01, 20.00)
#     sharpness_min_max = (0.06, 0.20)  
#     blur_motion_min_max = np.array([5, 7]) 
# ```
# We always perform a horizontal flip with probability 0.50, since we have seen it increases the performance of the model.
# 
# 
# ###### Step 4: Random search with different backbones and augmentations
# 
# A random search is performed with different backbones and values of the selected filters and within the ranges defined in step 2. This is done in several GPUs. Some models failed because they did not fit in some of the GPUs (Nvidia 3070), so sometimes we were not able to test each configuration.
# 
# ###### Step 5: Selection of the best models and use of pseudolabelling
# 
# After the random search, we have selected the models with the best scores. Such models are again trained using pseudo-labelling, i.e., using the prediction that has been previously obtained from the test dataset.
# 
# ###### Step 6: Ensemble
# 
# The best models of the previous step are selected and we build a Weighted Blending Ensemble to achieve a final XXXX f1-score. To do it, we tested the combinations of the 2, 3, 4, ..., N best models that have been obtained in the previous step, where N is optimized to get the best result. The weights of the models are obtained by optimization with the validation predictions.
# 
# 
# ---
# 
# #### Further comments:
# ---
# 
# The code can be used directly from the notebook or with its python version from the console. 
# 
# The following example allows one to train a "tf_efficientnet_b3_ns" model with a 3-fold cross-validation and several augmentations (horizontal flip, color saturation, rotarion, sharpness and motion blur): 
# 
# ```
# python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_FIRST/ --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0 --VERSION 01003 --NUM_EPOCHS 130 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1 --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.181641967391821 --COLORJITTER_SATURATION 0.0927803557147827 --ROTATION_PROB 0.181641967391821 --ROTATION_DEGREES 19.1576227380687 --SHARPNESS_PROB 0.181641967391821 --SHARPNESS_VALUE 0.140246520091931 --MOTIONBLUR_PROB 0.181641967391821 --MOTIONBLUR_KERNEL_SIZE 7
# ```
# 
# The parameters' description follows:
# 
# 
# *Training Parameters*
# 
# - OUTPUT_DIR = Directory for results
# - TRAIN_MODEL = To train models
# - CREATE_SUBMISSION = To create submissions
# - VERBOSE = Verbose
# - INCLUDE_TEST = Include test in train for pseudo-labeling
# - MIN_THR_TEST = Threshold of preds to include in pseudo
# - TARGET_DIR_PREDS = Directory with preds for test in pseudo
# - PREVIOUS_MODEL = Dir with previous model
# - BACKBONE = Backbone
# - GPU_DEVICE = GPU device
# - SEED = Main Seed
# - NUM_FOLDS = Number of folds
# - RUN_FOLDS = Number of folds to run (1 to quick validation)
# - LR = Min Learning Rate
# - NUM_EPOCHS = Max epochs
# - BATCH_SIZE = Batch size
# - NUM_WORKERS = Numworkers in Dataloader
# - SAMPLES_BY_CLASS = Number of row of each class included in balanced train db
# - USA_FP16 = To use 16 Float in GPU
# - VERSION = Code Version
# 
# [Kornia Augmentation Filters](https://kornia.readthedocs.io/en/v0.4.1/augmentation.html#module-api)
# 
# - AUGMENTATION = Apply augmentation
# - HORIZONTAL_FLIP = Probability for flip
# - VERTICAL_FLIP = Probability for flip
# - ERASING_MAX_PROB = Probability for erasing
# - ERASING_MAX_RATIO = Max ratio box to erase
# - COLORJITTER_PROB = Probability for colorjitter
# - COLORJITTER_BRIGHTNESS = Value for brightness
# - COLORJITTER_CONTRAST = Value for contrast
# - COLORJITTER_SATURATION = Value for saturation
# - COLORJITTER_HUE = Value for hue
# - ROTATION_PROB = Probability for rotation
# - ROTATION_DEGREES = Max degrees for rotation
# - SHARPNESS_PROB = Probability for sharpness
# - SHARPNESS_VALUE = Max value of sharpness
#       
# - MOTIONBLUR_PROB = Probability for motionblur
# - MOTIONBLUR_KERNEL_SIZE = Max size ofkernel motion
# - MOTIONBLUR_KERNEL_ANGLE = Max angle of the motion
# - MOTIONBLUR_KERNEL_DIRECTION = Direction of the motion

# # Import Packages

# In[44]:


import os, glob, random, time, sys, pickle, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ast

from scipy import signal
import cv2
import logging
from contextlib import contextmanager
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
import torch.utils.data as torchdata
# import torch.optim as optim
import torch_optimizer as optim
from torchvision import transforms, models, datasets, io
from torchvision.utils import make_grid

from transformers import get_linear_schedule_with_warmup
import transformers
from torch.cuda.amp import autocast, GradScaler

from albumentations.core.transforms_interface import ImageOnlyTransform
import kornia

from tqdm import tqdm
# from tqdm.notebook import tqdm
from functools import partial
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split, KFold
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix, mean_squared_error, precision_recall_fscore_support
from skimage.transform import resize

import albumentations as A
import albumentations.pytorch.transforms as T
from IPython.display import clear_output

from PIL import Image, ImageOps
import timm
from timm.models.efficientnet import tf_efficientnet_b0_ns, tf_efficientnet_b1_ns, tf_efficientnet_b2_ns, tf_efficientnet_b3_ns, tf_efficientnet_b4_ns
from timm.models.efficientnet import tf_efficientnet_b6_ns
from timm.models.mobilenetv3 import mobilenetv3_small_075, mobilenetv3_small_100, mobilenetv3_large_075, mobilenetv3_large_100
from timm.models.resnest import resnest50d, resnest101e, resnest200e

import warnings
# warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

torch.__version__, np.__version__, pd.__version__, kornia.__version__


# # Console Commands

# In[45]:


# ArgumentS From Console
# ----------------------
def in_nb():
    import __main__ as main
    return not hasattr(main, '__file__')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--OUTPUT_DIR", type=str, default="None") #Directory for results

# Training parameters
parser.add_argument("--TRAIN_MODEL", type=str, default="True") # To train models
parser.add_argument("--CREATE_SUBMISSION", type=str, default="True") # To create submissions
parser.add_argument("--VERBOSE", type=int, default=0) #Verbose?
parser.add_argument("--INCLUDE_TEST", type=str, default="False") # Include test in train for pseudo-labeling
parser.add_argument("--MIN_THR_TEST", type=float, default= -99999) # Threshold of preds to include in pseudo
parser.add_argument("--TARGET_DIR_PREDS", type=str, default="None") # Directory with preds for test in pseudo
parser.add_argument("--PREVIOUS_MODEL", type=str, default="None") # Dir with previous model
parser.add_argument("--BACKBONE", type=str, default="tf_efficientnet_b3_ns") #Backbone
parser.add_argument("--GPU_DEVICE", type=str, default="cuda:0") # GPU device
parser.add_argument("--SEED", type=int, default=12345) # Main Seed
parser.add_argument("--NUM_FOLDS", type=int, default=5) # Number of folds
parser.add_argument("--RUN_FOLDS", type=int, default=1) # Number of folds to run (1 to quick validation)
parser.add_argument("--LR", type=float, default=0.00001) # Min LR
parser.add_argument("--NUM_EPOCHS", type=int, default=40) # Max epochs
parser.add_argument("--BATCH_SIZE", type=int, default=16) # Batsize
parser.add_argument("--NUM_WORKERS", type=int, default=4) # Numworkers in Dataloader
parser.add_argument("--SAMPLES_BY_CLASS", type=int, default=64) # Number of row of each class included in balanced train db
parser.add_argument("--USA_FP16", type=str, default="False") # To use 16 Float in GPU
parser.add_argument("--VERSION", type=str, default="01002") #Version of code

# Augmentation https://kornia.readthedocs.io/en/v0.4.1/augmentation.html#module-api
parser.add_argument("--AUGMENTATION", type=str, default="False") #Apply augmentation
parser.add_argument("--HORIZONTAL_FLIP", type=float, default=0.0) #Probability for flip
parser.add_argument("--VERTICAL_FLIP", type=float, default=0.0) #Probability for flip

parser.add_argument("--ERASING_MAX_PROB", type=float, default=0.0) #Probability for erasing
parser.add_argument("--ERASING_MAX_RATIO", type=float, default=0.33) #Max ratio box to erase

parser.add_argument("--COLORJITTER_PROB", type=float, default=0.0) #Probability for colorjitter
parser.add_argument("--COLORJITTER_BRIGHTNESS", type=float, default=0.0) #Value for
parser.add_argument("--COLORJITTER_CONTRAST", type=float, default=0.0) #Value for
parser.add_argument("--COLORJITTER_SATURATION", type=float, default=0.0) #Value for
parser.add_argument("--COLORJITTER_HUE", type=float, default=0.0) #Value for

parser.add_argument("--ROTATION_PROB", type=float, default=0.0) #Probability for rotation
parser.add_argument("--ROTATION_DEGREES", type=float, default=0.10) #Max degrees for rotation

parser.add_argument("--SHARPNESS_PROB", type=float, default=0.0) #Probability for sharpness
parser.add_argument("--SHARPNESS_VALUE", type=float, default=0.10) #Max value of sharpness
      
parser.add_argument("--MOTIONBLUR_PROB", type=float, default=0.0) #Probability for motionblur
parser.add_argument("--MOTIONBLUR_KERNEL_SIZE", type=int, default=3) #Max size ofkernel motion
parser.add_argument("--MOTIONBLUR_KERNEL_ANGLE", type=float, default=0.10) #Max angle of the motion
parser.add_argument("--MOTIONBLUR_KERNEL_DIRECTION", type=float, default=0.0) #Direction of the motion


# In[46]:


COMMAND = 'CONSOLE'
# VERSION = "01002"
# COMMAND = 'BASIC'

if COMMAND=='BASIC':
    cmdline = "--OUTPUT_DIR results/BASIC_MODEL_008_HORIZONTAL_FLIP/"
    cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
    cmdline += " --NUM_EPOCHS 50 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 3"
    cmdline += " --AUGMENTATION True --HORIZONTAL_FLIP 0.50" #--COLORJITTER_PROB 0.50 --COLORJITTER_BRIGHTNESS 0.05"
    cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
#     cmdline += " --INCLUDE_TEST True --TARGET_DIR_PREDS results/BASIC_MODEL/"
    cmdline = cmdline.split(' ')

if COMMAND=='CONSOLE':
    cmdline = sys.argv[1:]   # Read from console

arg_in = parser.parse_args(cmdline)
str_todos = ' '.join(str(arg_in).split(' '))
# str_todos = str_todos.replace(',','__')
# str_todos = str_todos.replace('Namespace(','')
# str_todos = str_todos.replace(')','')
# str_todos = str_todos.replace('=','_')
# str_todos = str_todos.replace('.','_')
# str_todos = str_todos.replace("'","")
# str_todos = str_todos.replace('\\r','')
print(str_todos)


# In[ ]:


# Convert to boolean
arg_in.INCLUDE_TEST = True if arg_in.INCLUDE_TEST=='True' else False
arg_in.TRAIN_MODEL = True if arg_in.TRAIN_MODEL=='True' else False
arg_in.AUGMENTATION = True if arg_in.AUGMENTATION=='True' else False
arg_in.USA_FP16 = True if arg_in.USA_FP16=='True' else False
arg_in.CREATE_SUBMISSION = True if arg_in.CREATE_SUBMISSION=='True' else False


# # Config

# In[ ]:


# [i for i in timm.list_models(pretrained=True) if 'efficient' in i]


# In[ ]:


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.cuda.is_available()


# In[ ]:


OUTPUT_DIR = arg_in.OUTPUT_DIR
GPU_DEVICE = arg_in.GPU_DEVICE
VERSION = arg_in.VERSION

BACKBONE = arg_in.BACKBONE
AUGMENTATION = arg_in.AUGMENTATION
HORIZONTAL_FLIP = arg_in.HORIZONTAL_FLIP
VERTICAL_FLIP = arg_in.VERTICAL_FLIP
ERASING_MAX_PROB = arg_in.ERASING_MAX_PROB
ERASING_MAX_RATIO = arg_in.ERASING_MAX_RATIO

COLORJITTER_PROB = arg_in.COLORJITTER_PROB
COLORJITTER_BRIGHTNESS = arg_in.COLORJITTER_BRIGHTNESS
COLORJITTER_CONTRAST = arg_in.COLORJITTER_CONTRAST
COLORJITTER_SATURATION = arg_in.COLORJITTER_SATURATION
COLORJITTER_HUE = arg_in.COLORJITTER_HUE

ROTATION_PROB = arg_in.ROTATION_PROB
ROTATION_DEGREES = arg_in.ROTATION_DEGREES

SHARPNESS_PROB = arg_in.SHARPNESS_PROB
SHARPNESS_VALUE = arg_in.SHARPNESS_VALUE

MOTIONBLUR_PROB = arg_in.MOTIONBLUR_PROB
MOTIONBLUR_KERNEL_SIZE = arg_in.MOTIONBLUR_KERNEL_SIZE
MOTIONBLUR_KERNEL_ANGLE = arg_in.MOTIONBLUR_KERNEL_ANGLE
MOTIONBLUR_KERNEL_DIRECTION= arg_in.MOTIONBLUR_KERNEL_DIRECTION

# Train parameters
NUM_FOLDS = arg_in.NUM_FOLDS
RUN_FOLDS = arg_in.RUN_FOLDS
CREATE_SUBMISSION = arg_in.CREATE_SUBMISSION
PREVIOUS_MODEL = arg_in.PREVIOUS_MODEL
SEED = arg_in.SEED

TRAIN_MODEL = arg_in.TRAIN_MODEL
INCLUDE_TEST = arg_in.INCLUDE_TEST
MIN_THR_TEST = arg_in.MIN_THR_TEST
TARGET_DIR_PREDS = arg_in.TARGET_DIR_PREDS
USA_FP16 = arg_in.USA_FP16
NUM_CLASSES = 3
device = (GPU_DEVICE if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


# Training parameters
BATCH_TRAIN = arg_in.BATCH_SIZE #16
NUM_WORKERS = arg_in.NUM_WORKERS #4
BATCH_VALID = BATCH_TRAIN
BATCH_TEST = BATCH_TRAIN

# --------------------------------------
# Define 'CosineAnnealingWarmUpRestarts' parameters
LR = arg_in.LR #1e-5 #Learning rate
T_MULT = 1 # Multiplier factor of next cycle
ETA_MAX = 0.001  #0.002 0.001 # Maximum value of LR
EPOCHS_UP = 5 # Epochs up cicle  EPOCHS_CYCLE//4
NUM_EPOCHS = arg_in.NUM_EPOCHS #40 # Max epochs
EPOCHS_CYCLE = NUM_EPOCHS # Epochs in a complete cycle  EPOCHS//(1+T_MULT+(T_MULT*T_MULT))

SAMPLES_BY_CLASS = arg_in.SAMPLES_BY_CLASS #64
ITERS_EPOCH = int(SAMPLES_BY_CLASS * NUM_CLASSES // BATCH_TRAIN) #*(NUM_FOLDS-1) // (BATCH_TRAIN*(NUM_FOLDS)) #Calcula los iters por epoch
print('ITERS_EPOCH=', ITERS_EPOCH, 'EPOCHS_UP=', EPOCHS_UP, 'NUM_EPOCHS=', NUM_EPOCHS)
GAMMA = 0.90 # Reduction factor in each cycle
# -------------------------------------
EARLY_STOP = 9999 # Early stop


# # Utils

# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def mean_auc(y_true, y_pred):
    auc_mean = []
    for ncol in range(y_true.shape[1]):
        if np.sum(y_true[:,ncol])>0:
            auc_mean.append(roc_auc_score(y_true[:,ncol], y_pred[:, ncol]))
    return np.mean(auc_mean)


# In[ ]:


# https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=(15,10), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# In[ ]:


# Use from jupyter notebook notebook
from notebook import notebookapp
import urllib
import json
import ipykernel
from pathlib import Path

def notebook_path():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
        except:
            pass  # There may be stale entries in the runtime directory 
    return None

def create_output_dir(display = True, create=True, dirbase = '../results'):
    base_dir = Path(dirbase)
    if not os.path.exists(base_dir) and create:
        os.mkdir(base_dir)
    currentNotebook = notebook_path().split('/')
    currentNotebook = currentNotebook[-1]
    dir_models = currentNotebook[:-6]
    dir_models = dir_models.replace('.','_')
    output_dir = base_dir / Path(dir_models) #settings["globals"]["output_dir"])

    if not os.path.exists(output_dir) and create:
        os.mkdir(output_dir)
        if display:
            print("Directory " , output_dir ,  " Created ")
    elif display:
        print("Directory " , output_dir ,  " already exists")
    return output_dir


# In[47]:


import math
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# # Create DataSet

# In[48]:


# Create Directory
if OUTPUT_DIR == 'None':
    output_dir = create_output_dir(display=True, create=True, dirbase = 'results/')
else:
    output_dir = Path(OUTPUT_DIR)

if not os.path.exists(output_dir):
    if not os.path.exists('results'):
        os.mkdir('results')
    os.mkdir(output_dir)
    print("Directory " , output_dir ,  " Created ")


# In[ ]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
len_train_df = len(train_df)
train_df.label.unique()


# ## Folds in Train

# In[ ]:


seed_everything(SEED)
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_df['kfold'] = -1
for fold, (t_idx, v_idx) in enumerate(kfold.split(train_df, y=train_df.label)):
    train_df.loc[v_idx, "kfold"] = fold
train_df.head()

for fold in range(NUM_FOLDS):
    print('\FOLD DISTRIBUTION:',fold, list(train_df[train_df.kfold==fold]['label'].value_counts()))


# In[ ]:


train_df.head(), test_df.head()


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


train_df['label'].value_counts()


# ## Include Test for PseudoLabelling

# In[49]:


if INCLUDE_TEST:
    print(f'Training with train+test datasets using {TARGET_DIR_PREDS} predictions.')
    test_df_tmp = test_df.copy()
    
    with open(Path(TARGET_DIR_PREDS) / f'preds_test.pkl', 'rb') as file:
        preds_test = pickle.load(file)
    
    # Select only rows greather than MIN_THR_TEST
    max_in_row = np.max(preds_test, 1)
    selec_rows_test = max_in_row > MIN_THR_TEST
    test_df_tmp['label'] = np.argmax(preds_test, 1)
    test_df_tmp = test_df_tmp.loc[selec_rows_test].reset_index(drop=True)
    
    # Include kfold columns
    seed_everything(SEED)
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    test_df_tmp['kfold'] = -1
    for fold, (t_idx, v_idx) in enumerate(kfold.split(test_df_tmp, y=test_df_tmp.label)):
        test_df_tmp.loc[v_idx, "kfold"] = fold
    test_df_tmp = test_df_tmp[train_df.columns].reset_index(drop=True)
    train_df = pd.concat([train_df, test_df_tmp]).reset_index(drop=True)

print(train_df.shape, test_df.shape)


# # Look a Sample

# In[ ]:


if False:
    imgs = []
    labels = []
    for nrow, row in train_df.iloc[:6].iterrows():
        img = Image.open(row['example_path'])
        img = transforms.PILToTensor()(img)
        imgs.append(img)
        labels.append(row['label'])
    grid = make_grid(imgs, nrow=3)
    show(grid)
    plt.title(labels)
    
    imgs = []
    for nrow, row in test_df.iloc[:6].iterrows():
        img = Image.open(row['example_path'])
        img = transforms.PILToTensor()(img)
        imgs.append(img)
    grid = make_grid(imgs, nrow=3)
    show(grid)


# # Augmentations

# In[ ]:


# See: https://kornia.readthedocs.io/en/v0.4.1/augmentation.html#module-api
def augment_imgs(imgs):
    if HORIZONTAL_FLIP>0.0:
        imgs = kornia.augmentation.RandomHorizontalFlip(p=HORIZONTAL_FLIP)(imgs)
    if VERTICAL_FLIP>0.0:
        imgs = kornia.augmentation.RandomVerticalFlip(p=VERTICAL_FLIP)(imgs)
    if ERASING_MAX_PROB>0.0:
        imgs = kornia.augmentation.RandomErasing(ratio=(0.02, ERASING_MAX_RATIO), p=ERASING_MAX_PROB)(imgs)
    if COLORJITTER_PROB>0.0:
        imgs = kornia.augmentation.ColorJitter(brightness=COLORJITTER_BRIGHTNESS,
                                            contrast=COLORJITTER_CONTRAST,
                                            saturation=COLORJITTER_SATURATION,
                                            hue=COLORJITTER_HUE,
                                            p=COLORJITTER_PROB)(imgs)
    if ROTATION_PROB>0.0:
        imgs = kornia.augmentation.RandomRotation(degrees=ROTATION_DEGREES,
                                                  p=ROTATION_PROB)(imgs)
        
    if SHARPNESS_PROB>0.0:
        imgs = kornia.augmentation.RandomSharpness(sharpness=SHARPNESS_VALUE,
                                                  p=SHARPNESS_PROB)(imgs)
        
    if MOTIONBLUR_PROB>0.0:
        imgs = kornia.augmentation.RandomMotionBlur(kernel_size=MOTIONBLUR_KERNEL_SIZE,
                                                    angle=MOTIONBLUR_KERNEL_ANGLE,
                                                    direction=MOTIONBLUR_KERNEL_DIRECTION,
                                                    p=MOTIONBLUR_PROB)(imgs)
    return imgs


# # DataLoaders

# In[50]:


class ImgDataset:
    def __init__(self, df, transforms=None, mode="train"):
        self.mode = mode
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        filename = record['example_path']
#         img = Image.open(filename)
#         img = transforms.PILToTensor()(img)
        img = cv2.imread(filename)
        img = kornia.image_to_tensor(img)
        img = kornia.color.bgr_to_rgb(img)
        img = img.type(torch.DoubleTensor) / 255.
        labels = np.zeros(3)
        if self.mode=='train' or self.mode=='valid':
            labels[record['label']] = 1.0
        return {"img" : img, "labels" : labels, 'filename' : filename}
    
# # Test Visualize img and labels
# # -----------------------------------
# seed_everything(1234)
# trainset = ImgDataset(df = train_df.iloc[:8], mode="train")
# train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True, num_workers=0)
# for input_dl in tqdm(train_loader):
#     if True:
#         input_dl['img'] = transform_kornia(input_dl['img'].to(device))
# #     print(input_dl['img'].shape)
# #     print(input_dl['labels'].cpu().numpy()[0])
    
#     fig, ax = plt.subplots(1,2,figsize=(20,10))
#     img0 = kornia.tensor_to_image(input_dl['img'][0])
#     img1 = kornia.tensor_to_image(input_dl['img'][1])
#     ax[0].imshow(img0)
#     ax[0].axis('off')    
#     ax[0].set_title('Label=' + str(input_dl['labels'][0].numpy()) + ' Name=' + input_dl['filename'][0].split('/')[-1])
#     ax[1].imshow(img1)
#     ax[1].axis('off')
#     ax[1].set_title('Label=' + str(input_dl['labels'][1].numpy()) + ' Name=' + input_dl['filename'][1].split('/')[-1])
#     plt.show()


# # Model

# In[51]:


def Model():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=NUM_CLASSES, in_chans=3)
    return model


# # Train/Val Funs

# In[52]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[53]:


# For training
def train_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    model.train()
    y_true = []
    y_pred = []
    losses = AverageMeter()
    
    t = tqdm(loader)
    for i, sample in enumerate(t):
        optimizer.zero_grad()
        input_img = sample['img'].float().to(device)
        
        if AUGMENTATION:
            input_img = augment_imgs(input_img)
            
        target = sample['labels'].float().to(device)
        bs = input_img.size(0)
        with torch.cuda.amp.autocast():
#             output = model(input)
#             loss = criterion(output['logit'], target) #BCEWithLogitLoss

            output = model(input_img)
            max_label = torch.argmax(target, dim=1)
            loss = criterion(output, max_label)   # CrossEntropyLoss

        if USA_FP16:
            # For FP16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
        
        y_true.append(target.cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
        losses.update(loss.item(), bs)
        t.set_description(f"Train E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return losses.avg, y_true, y_pred

# For validation
def valid_epoch(model, loader, criterion, epoch):
    model.eval()
    y_true = []
    y_pred = []
    losses = AverageMeter()
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input_img = sample['img'].float().to(device)
            target = sample['labels'].float().to(device)
            bs = input_img.size(0)
            
#             output = model(input)
#             loss = criterion(output['logit'], target) #BCEWithLogitLoss
            
            output = model(input_img)
            max_label = torch.argmax(target,dim=1)
            loss = criterion(output, max_label)
            
            y_true.append(target.cpu().numpy())
            y_pred.append(output.detach().cpu().numpy())
            losses.update(loss.item(), bs)
            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
        t.close()
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return losses.avg, y_true, y_pred


# # Training

# In[26]:


# # GPU con 8 Gigas
# BATCH_TRAIN = 16 #10
# NUM_WORKERS = 4
# BATCH_VALID = BATCH_TRAIN
# BATCH_TEST = BATCH_TRAIN

# # --------------------------------------
# # Parametros para optimizar con 'CosineAnnealingWarmUpRestarts'
# LR = 1e-5 #Learning rate
# T_MULT = 1 # Factor multiplicador de  la longitud del siguiente ciclo
# ETA_MAX = 0.001  #0.002 0.001 # Valor maximo LR
# EPOCHS_UP = 5 # Epochs ciclo de subida  EPOCHS_CYCLE//4
# NUM_EPOCHS = 40 # Max epochs
# EPOCHS_CYCLE = NUM_EPOCHS #Epochs ciclo completo EPOCHS//(1+T_MULT+(T_MULT*T_MULT))
# # Num iters per epoch len of (num_folds-1) trainning DB / BATCH_SIZE

# SAMPLES_BY_CLASS = 64
# AUGMENTATION = False

# ITERS_EPOCH = int(SAMPLES_BY_CLASS * NUM_CLASSES // BATCH_TRAIN) #*(NUM_FOLDS-1) // (BATCH_TRAIN*(NUM_FOLDS)) #Calcula los iters por epoch
# # ITERS_EPOCH = int(len(df_para_sed.query('kfold > 2')) // (BATCH_TRAIN)) #Calcula los iters por epoc
# print(ITERS_EPOCH, EPOCHS_UP, NUM_EPOCHS)

# GAMMA = 0.90 # Factor reductor del maximo de cada ciclo
# # -------------------------------------
# EARLY_STOP = 9999 # Numero de epochs para si no mejora


# output_dir = create_output_dir(display=True, create=True, dirbase = 'resultados')
# # output_dir = Path('resultados/0049_model/')
# str(output_dir)


# In[27]:


if TRAIN_MODEL:
    start = time.time()
    trues_total = []
    preds_total = []
    for fold in np.arange(RUN_FOLDS): 
        print('FOLD:',fold)
        # Only valid fold from original train
        valid_df_fold = train_df.iloc[:len_train_df].reset_index(drop=True)
        valid_df_fold = valid_df_fold.loc[valid_df_fold['kfold'] == fold].reset_index(drop=True)
        train_df_fold = train_df[train_df['kfold'] != fold].reset_index(drop=True)
        print(train_df_fold.shape, valid_df_fold.shape)
        
        model = Model()
        if PREVIOUS_MODEL!='None':
            model.load_state_dict(torch.load(f'{PREVIOUS_MODEL}modelo_fold-{fold}.bin'))
        model = model.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # https://pythonawesome.com/a-collection-of-optimizers-for-pytorch/
        optimizer = optim.DiffGrad(model.parameters(), lr=LR)
        if USA_FP16:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=ITERS_EPOCH*EPOCHS_CYCLE, T_mult=T_MULT, 
                                          eta_max=ETA_MAX, T_up=ITERS_EPOCH*EPOCHS_UP, gamma=GAMMA)
        # scheduler = None
        criterion = nn.CrossEntropyLoss()
#         criterion = nn.BCEWithLogitsLoss() # Multilabel

    
        # train loader
        # ------------
        trainset = ImgDataset(df=train_df_fold, mode="train")
        train_loader = DataLoader(trainset, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False) #, num_workers=NUM_WORKERS)
            
        # valid loader
        # ------------
        validset = ImgDataset(df=valid_df_fold, mode="valid")
        valid_loader = DataLoader(validset, batch_size=BATCH_VALID, shuffle=False, drop_last=False) #, num_workers=NUM_WORKERS)

        # Inicializamos variables
        # -----------------------
        best_auc = 0
        best_acc = 0
        best_f1_micro = 0
        best_f1_macro = 0
        best_f1_weighted = 0
        best_epoch = 0
        best_loss = np.inf
        early_stop_count = 0
        res = []
        best_preds_fold = []
        best_trues_fold = []  
        best_thr = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Bucle principal
        # ---------------
        
        for epoch in range(NUM_EPOCHS):
            seed_everything(SEED*(epoch+1))
            
            # use balanced train dataset
            # 0    860
            # 2    658
            # 1    196
            ttt1 = train_df_fold.query('label==0').sample(SAMPLES_BY_CLASS, replace=True).reset_index(drop=True)
            ttt2 = train_df_fold.query('label==1').sample(SAMPLES_BY_CLASS, replace=True).reset_index(drop=True)
            ttt3 = train_df_fold.query('label==2').sample(SAMPLES_BY_CLASS, replace=True).reset_index(drop=True)
            trainset = ImgDataset(df=pd.concat([ttt1,ttt2,ttt3]).reset_index(drop=True), mode="train")
            train_loader = DataLoader(trainset, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False)

            start_epoch = time.time()
#             warnings.filterwarnings('ignore')
            train_loss, y_true_train, y_pred_train = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
#             warnings.filterwarnings('ignore')
            valid_loss, y_true_valid, y_pred_valid = valid_epoch(model, valid_loader, criterion, epoch)
            
            
            final_loss = criterion(torch.tensor(y_pred_valid), torch.argmax(torch.tensor(y_true_valid),dim=1))
            valid_auc = mean_auc(y_true_valid, y_pred_valid)
            valid_acc = np.mean(np.argmax(y_true_valid,1)==np.argmax(y_pred_valid,1))
            f1_micro = metrics.f1_score(np.argmax(y_true_valid,1), np.argmax(y_pred_valid,1), average='micro')
            f1_macro = metrics.f1_score(np.argmax(y_true_valid,1), np.argmax(y_pred_valid,1), average='macro')
            f1_weighted = metrics.f1_score(np.argmax(y_true_valid,1), np.argmax(y_pred_valid,1), average='weighted')
            
            # Save the best model and results
            # --------------------------------
            if f1_macro > best_f1_macro:
                print(f"########## >>>>>>>> Model Improved From F1-macro={best_f1_macro} ----> loss=={f1_macro}")
                # print(f"########## >>>>>>>> Model Mejorado de loss={best_auc} ----> loss=={valid_auc}")
                torch.save(model.state_dict(), output_dir / f'modelo_fold-{fold}.bin')
                best_epoch = epoch
                best_loss = valid_loss
                best_acc = valid_acc
                best_auc = valid_auc
                best_f1_micro = f1_micro
                best_f1_macro = f1_macro
                best_f1_weighted = f1_weighted
                best_thr_all = best_thr.copy()
                early_stop_count = 0
                best_preds_fold = y_pred_valid
                best_trues_fold = y_true_valid
            else:
                early_stop_count += 1

                
            tiempo = round(((time.time() - start)/60),2)
            tiempo_epoch = round(((time.time() - start_epoch)/60),2)
#             clear_output(wait=True) #to clean warnings
            print(f'Fold={fold:02d} Epoch={epoch:02d} TrainLOSS={train_loss:.04f}\tValidLOSS={valid_loss:.04f}  ValidACC={valid_acc:.04f}  ValidAUC={valid_auc:0.4f}')
            print(f'\t\t\t f1_micro={f1_micro:.04f}  f1_macro={f1_macro:.04f}  f1_weighted={f1_weighted:0.4f}')
            print(f'BestE={best_epoch:02d} min={tiempo_epoch:.01f}/{tiempo:.01f}\tBest_LOSS={best_loss:.04f}  Best_ACC={best_acc:.04f}  Best_AUC={best_auc:.04f}')
            print(f'\t\t\t best_f1_micro={best_f1_micro:.04f}  best_f1_macro={best_f1_macro:.04f}  best_f1_weighted={best_f1_weighted:0.4f}')


            # Guarda resultados del epoch
            # ---------------------------
            if scheduler.__class__ ==  torch.optim.lr_scheduler.OneCycleLR:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']

            res.append(dict({'fold':fold, 'epoch':epoch, 'lr':lr, 'tiempo':tiempo,
                             'trn_loss':train_loss, 'val_loss':valid_loss, 
                             'val_acc':valid_acc, 'val_auc':valid_auc,
                             'f1_micro':f1_micro, 'f1_macro':f1_macro, 'f1_weighted':f1_weighted,
                             'best_epoch':best_epoch, 'best_loss':best_loss,
                             'best_acc':best_acc, 'best_auc':best_auc,
                             'best_f1_micro':best_f1_micro, 'best_f1_macro':best_f1_macro, 'best_f1_weighted':best_f1_weighted,
                             }))
            res_df = pd.DataFrame(res)
            res_df.to_csv(output_dir / f'modelo_fold-{fold}.csv')

            # Guarda predicciones
            # -------------------
            with open(output_dir / f'modelo_fold-{fold}.pkl', 'wb') as file:
                pickle.dump(best_preds_fold, file)
                pickle.dump(best_trues_fold, file)

            if COMMAND!='CONSOLE':
                # Draw curves if not in console
                # -----------------------------
                fig, axs = plt.subplots(2,2, figsize=(15,15))
                axs[0,0].plot(res_df['trn_loss'].values, label='trn_loss')
                axs[0,0].plot(res_df['val_loss'].values, label='val_loss')
                axs[0,0].set_xlabel('Epochs')
                axs[0,0].set_ylabel('Loss')
                axs[0,0].set_title(f'Val_loss={valid_loss:.6f} Best={best_loss:.6f} in Epoch{best_epoch}')
                axs[0,0].legend(loc='upper right')

                axs[0,1].plot(res_df['val_acc'].values, label='val_acc')
    #             axs[0,1].plot(res_df['f1_micro'].values, label='f1_micro')
                axs[0,1].plot(res_df['f1_macro'].values, label='f1_macro')
        #         axs[0,1].plot(res_df['f1_weighted'].values, label='f1_weighted')
                axs[0,1].set_xlabel('Epochs')
                axs[0,1].set_ylabel('Acc y f1macro')
                axs[0,1].set_title(f'ACC={valid_acc:.4f} fmac={f1_macro:.5f} Best:[ACC={best_acc:.4f} fmac={best_f1_macro:.5f}')
                axs[0,1].legend(loc='lower right')

                axs[1,0].plot(res_df['val_auc'].values, label='auc')
                axs[1,0].set_xlabel('Epochs')
                axs[1,0].set_ylabel('AUC MEAN')
                axs[1,0].set_title(f'AUC={valid_auc:.6f} BestAUC={best_auc:.6f} in Epoch{best_epoch}')

                axs[1,1].plot(res_df['lr'].values)
                axs[1,1].set_xlabel('Epochs')
                axs[1,1].set_ylabel('Learning Rate')
                axs[1,1].set_title(f'Learning Rate={lr:.8f} Max={res_df.lr.max():.8f} Min={res_df.lr.min():.8f}')
                fig.savefig(output_dir / f'modelo_fold-{fold}.png',facecolor='white', edgecolor='white')
                plt.close(fig)

            # Si se alcanza parada temprana sal
            # ---------------------------------
            if EARLY_STOP == early_stop_count:
                print('\n !!! ALCANZADO EARLY STOPPING EN EL EPOCH:', epoch, '!!! MEJOR MODELO EN EPOCH:', best_epoch)
                break
        trues_total.append(best_trues_fold)
        preds_total.append(best_preds_fold)

    best_trues_fold = np.vstack(trues_total)
    best_preds_fold = np.vstack(preds_total)
    FINAL_F1MACRO =  metrics.f1_score(np.argmax(best_trues_fold,1), np.argmax(best_preds_fold,1), average='macro')
    FINAL_ACC = np.mean(np.argmax(best_trues_fold,1)==np.argmax(best_preds_fold,1))
    print('\n\nBEST FINAL F1-MACRO=', FINAL_F1MACRO, "ACC=", FINAL_ACC)


# # Crea Submission

# In[28]:


if CREATE_SUBMISSION:
    print(f'\n\nCREATING SUBMISSION {output_dir} WITH F1-MACRO={FINAL_F1MACRO} and ACC={FINAL_ACC}')

    # Obtrain preds with the models
    preds_test = []
    testset = ImgDataset(df=test_df, mode="test")
    test_loader = DataLoader(testset, batch_size=BATCH_VALID, shuffle=False, drop_last=False)
    for fold in np.arange(RUN_FOLDS): 
        model = Model()
        model.load_state_dict(torch.load(output_dir / f'modelo_fold-{fold}.bin'))
        model = model.to(device)
        _, _, y_pred_tst = valid_epoch(model, test_loader, criterion, -1)
        preds_test.append(y_pred_tst)
    preds_test = np.stack(preds_test)
    preds_test = np.mean(preds_test, axis=0)
    with open(output_dir / f'preds_test.pkl', 'wb') as file:
        pickle.dump(preds_test, file)
        
    # Create Submission
    submission=test_df.copy()[['example_path']]
    submission['target']= submission['example_path'].str.split('/').str[-1].str.strip('.png')
    submission['label'] = np.argmax(preds_test, 1)
    submission = submission.drop(columns=['example_path'])
    submission = submission.astype({'label':'int','target':'int'})
    submission=submission.sort_values(by=['target'])
    print(submission.head())
    submission.to_csv(output_dir / 'submission.csv',index=False)
    submission.set_index('target').rename(columns={"label": "target"}).to_json(output_dir / 'submission.json',orient="columns")
    
# Save results in a csv
res_dict = {**dict(F1MACRO=np.round(FINAL_F1MACRO,6), ACC=np.round(FINAL_ACC,6)), **vars(arg_in)}
res_dict = pd.DataFrame(res_dict, index=[0])
name_resultados = f'results/resuls_console_v{VERSION}.csv'
if not os.path.exists(name_resultados):
    res_dict.to_csv(name_resultados, index=False)
else:
    resultados = pd.read_csv(name_resultados)
    res_dict = pd.concat([resultados, res_dict]).reset_index(drop=True)
    res_dict.to_csv(name_resultados, index=False)

print('LAST TWO ROWS...')
print(res_dict.tail(2).iloc[:,:10])
    
    


# # Scripts

# ## Scripts for searching best agumentations

# In[ ]:


if False:
    with open(f"colorjitter_brightness.sh", mode="w") as f:
        for brightness in np.linspace(0.001,0.20,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_BRIGHTNESS{brightness}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.50 --COLORJITTER_BRIGHTNESS {brightness}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"colorjitter_contrast.sh", mode="w") as f:
        for contrast in np.linspace(0.001,0.20,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_CONTRAST{contrast}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.50 --COLORJITTER_CONTRAST {contrast}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"colorjitter_saturation.sh", mode="w") as f:
        for saturation in np.linspace(0.001,0.20,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_SATURATION{saturation}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.50 --COLORJITTER_SATURATION {saturation}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"colorjitter_hue.sh", mode="w") as f:
        for hue in np.linspace(0.001,0.20,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_HUE{hue}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --COLORJITTER_PROB 0.50 --COLORJITTER_HUE {hue}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"erasing.sh", mode="w") as f:
        for erasing in np.linspace(0.04,0.25,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_ERASING{erasing}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --ERASING_MAX_PROB 0.50 --ERASING_MAX_RATIO {erasing}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"rotation.sh", mode="w") as f:
        for rotation in np.linspace(0.01,0.25,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_ROTATION{rotation}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --ROTATION_PROB 0.50 --ROTATION_DEGREES {rotation}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"sharpness.sh", mode="w") as f:
        for sharpness in np.linspace(0.001,0.25,11):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_SHARPNESS{sharpness}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:1"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --SHARPNESS_PROB 0.50 --SHARPNESS_VALUE {sharpness}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


if False:
    with open(f"motionblur.sh", mode="w") as f:
        for kernelsize in np.arange(5,20,2):
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR results/MODEL_23_MOTIONBLUR_KERNEL_SIZE{kernelsize}/"
            cmdline += " --VERBOSE 0 --BACKBONE tf_efficientnet_b3_ns --GPU_DEVICE cuda:0"
            cmdline += " --NUM_EPOCHS 100 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50 --MOTIONBLUR_PROB 0.50 --MOTIONBLUR_KERNEL_SIZE {kernelsize}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
    f.close()


# In[ ]:


# [i for i in timm.list_models(pretrained=True) if 'seresnext' in i]


# ## Scripts for random search of best models

# In[11]:


# Random Search Best Models
if False:
    backbones_list = np.array(['tf_efficientnet_b2_ns',
     'tf_efficientnet_b3_ns',
     'tf_efficientnet_b4_ns',
     'tf_efficientnet_b5_ns',
     'tf_efficientnet_b6_ns',
     'tf_efficientnet_b7_ns',
     'ig_resnext101_32x16d',
     'ig_resnext101_32x8d',
#      'tresnet_l_448',
#      'tresnet_m_448',
#      'tresnet_xl_448',
#      'vit_base_patch8_224',
#      'vit_base_patch16_384',
#      'vit_base_patch32_384',
     'resnest14d',
     'resnest26d',
     'resnest50d',
     'resnest101e',
     'resnest200e',
     'seresnext26d_32x4d',
     'seresnext50_32x4d',
     'seresnext101d_32x8d'])

    brightness_min_max = (0.07, 0.09)
    contrast_min_max = (0.07, 0.09) # 
    saturation_min_max = (0.01, 0.20)
    hue_min_max = (0, 0.1)
    erasing_min_max = (0.07, 0.09) # 
    rotation_min_max = (0.01, 20.00)
    sharpness_min_max = (0.06, 0.20) # 
    blur_motion_min_max = np.array([5, 7]) #np.arange(5,8,2) #Must be odd and >3

    random.seed(4567)
    np.random.seed(4567)
    for name_server in ['gru0', 'gru1', 'e29', 'e30', 'e31', 'ramon']:
        with open(f"best_backbone_server_{name_server}_NUEVO.sh", mode="w") as f:
            for num_model in np.arange(500): #len(backbones_list)):
                prob_augment = np.random.uniform(low=0.10, high=0.30)
                backbone = np.random.choice(backbones_list) #backbones_list[num_model] #

                brightness = np.random.uniform(low=brightness_min_max[0], high=brightness_min_max[1])
                contrast = np.random.uniform(low=contrast_min_max[0], high=contrast_min_max[1])
                saturation = np.random.uniform(low=saturation_min_max[0], high=saturation_min_max[1])
                hue = np.random.uniform(low=hue_min_max[0], high=hue_min_max[1])
                erasing = np.random.uniform(low=erasing_min_max[0], high=erasing_min_max[1])
                rotation = np.random.uniform(low=rotation_min_max[0], high=rotation_min_max[1])
                sharpness = np.random.uniform(low=sharpness_min_max[0], high=sharpness_min_max[1])
                kernelsize = np.random.choice(blur_motion_min_max)

                gpu_device = 'cuda:0' if name_server!='gru1' else 'cuda:1'
                output_dir_script = "results/MODEL_BEST_BACKBONE/" if name_server!='gru1' else "results/MODEL_BEST_BACKBONE_GRU1/"
                cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR {output_dir_script}"
                cmdline += f" --VERBOSE 0 --BACKBONE {backbone} --GPU_DEVICE {gpu_device} --VERSION 01003"
                cmdline += " --NUM_EPOCHS 130 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
#                 cmdline += " --NUM_EPOCHS 2 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
                cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50"
                cmdline += f" --COLORJITTER_PROB {prob_augment}"
    #             cmdline += f" --COLORJITTER_BRIGHTNESS {brightness}"
#                 cmdline += f" --COLORJITTER_CONTRAST {contrast}"
                cmdline += f" --COLORJITTER_SATURATION {saturation}"
#                 cmdline += f" --COLORJITTER_HUE {hue}"
#                 cmdline += f" --ERASING_MAX_PROB {prob_augment} --ERASING_MAX_RATIO {erasing}"
                cmdline += f" --ROTATION_PROB {prob_augment} --ROTATION_DEGREES {rotation}"
                cmdline += f" --SHARPNESS_PROB {prob_augment} --SHARPNESS_VALUE {sharpness}"
                cmdline += f" --MOTIONBLUR_PROB {prob_augment} --MOTIONBLUR_KERNEL_SIZE {kernelsize}"
                cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
                cmdline += "\n"
                f.write(cmdline)
        f.close()


# ## Scripts for training the best models with Pseudolabeling

# In[3]:


if True:
    best_models_df = pd.read_csv('results/the_best_models_final.csv')
    best_models_df = best_models_df.sort_values('F1MACRO', ascending=False)
    name_server_list = ['gru0', 'gru1', 'e29', 'e30', 'e31', 'ramon']
    for nrow, row in best_models_df.head(12).iterrows():
        name_server = name_server_list[nrow % 6]
        prob_augment = row['COLORJITTER_PROB']
        backbone = row['BACKBONE']
        saturation = row['COLORJITTER_SATURATION']
        rotation = row['ROTATION_DEGREES']
        sharpness = row['SHARPNESS_VALUE']
        kernelsize = row['MOTIONBLUR_KERNEL_SIZE']
        output_dir_script = f"results/BESTMODEL{nrow}/"
        output_dir_script_test = f"results/BESTMODEL{nrow}_TEST/"
        gpu_device = 'cuda:0' if name_server!='gru1' else 'cuda:1'

        with open(f"Final_model_{name_server}_num_model{nrow}.sh", mode="w") as f:
            # Original Model
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR {output_dir_script}"
            cmdline += f" --VERBOSE 0 --BACKBONE {backbone} --GPU_DEVICE {gpu_device} --VERSION 01004"
            cmdline += " --NUM_EPOCHS 130 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 5 --RUN_FOLDS 5"
    #                 cmdline += " --NUM_EPOCHS 2 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50"
            cmdline += f" --COLORJITTER_PROB {prob_augment}"
    #             cmdline += f" --COLORJITTER_BRIGHTNESS {brightness}"
    #                 cmdline += f" --COLORJITTER_CONTRAST {contrast}"
            cmdline += f" --COLORJITTER_SATURATION {saturation}"
    #                 cmdline += f" --COLORJITTER_HUE {hue}"
    #                 cmdline += f" --ERASING_MAX_PROB {prob_augment} --ERASING_MAX_RATIO {erasing}"
            cmdline += f" --ROTATION_PROB {prob_augment} --ROTATION_DEGREES {rotation}"
            cmdline += f" --SHARPNESS_PROB {prob_augment} --SHARPNESS_VALUE {sharpness}"
            cmdline += f" --MOTIONBLUR_PROB {prob_augment} --MOTIONBLUR_KERNEL_SIZE {kernelsize}"
            cmdline += " --INCLUDE_TEST False --TARGET_DIR_PREDS None"
            cmdline += "\n"
            f.write(cmdline)
        
            # Original Model with Pseudo-labelling
            cmdline = f"python 01006_FINAL_CODE.py --OUTPUT_DIR {output_dir_script_test}"
            cmdline += f" --VERBOSE 0 --BACKBONE {backbone} --GPU_DEVICE {gpu_device} --VERSION 01004"
            cmdline += " --NUM_EPOCHS 130 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 5 --RUN_FOLDS 5"
    #                 cmdline += " --NUM_EPOCHS 2 --BATCH_SIZE 16 --LR 0.00001 --NUM_FOLDS 3 --RUN_FOLDS 1"
            cmdline += f" --AUGMENTATION True --HORIZONTAL_FLIP 0.50"
            cmdline += f" --COLORJITTER_PROB {prob_augment}"
    #             cmdline += f" --COLORJITTER_BRIGHTNESS {brightness}"
    #                 cmdline += f" --COLORJITTER_CONTRAST {contrast}"
            cmdline += f" --COLORJITTER_SATURATION {saturation}"
    #                 cmdline += f" --COLORJITTER_HUE {hue}"
    #                 cmdline += f" --ERASING_MAX_PROB {prob_augment} --ERASING_MAX_RATIO {erasing}"
            cmdline += f" --ROTATION_PROB {prob_augment} --ROTATION_DEGREES {rotation}"
            cmdline += f" --SHARPNESS_PROB {prob_augment} --SHARPNESS_VALUE {sharpness}"
            cmdline += f" --MOTIONBLUR_PROB {prob_augment} --MOTIONBLUR_KERNEL_SIZE {kernelsize}"
            cmdline += f" --INCLUDE_TEST True --TARGET_DIR_PREDS {output_dir_script}"
            cmdline += "\n"
            f.write(cmdline)
        f.close()
        
    


# # Ensembling

# ## Load preds from the best models

# In[125]:


best_models_to_ensemble = [f"best_models/BESTMODEL{i}_TEST/" for i in [0,1,2,3,4,5,6,7,8,9,11]] # [1, 2, 6, 3, 11, 8, 5, 7, 0]] #[0,1,2,3,5,6,7,8,9,11]] #np.arange(12)]
best_models_to_ensemble


# In[126]:


test_preds_list = []
valid_preds_list = []
valid_trues_list = []
for dir_model in best_models_to_ensemble:
    # Valid preds
    trues_total = []
    preds_total = []
    for fold in np.arange(5):
        with open(f'{dir_model}modelo_fold-{fold}.pkl', 'rb') as file:
            best_preds_fold = pickle.load(file)
            best_trues_fold = pickle.load(file)
        trues_total.append(best_trues_fold)
        preds_total.append(best_preds_fold)
        
    preds_total = np.vstack(preds_total)
    trues_total = np.vstack(trues_total)
    
    FINAL_F1MACRO =  metrics.f1_score(np.argmax(trues_total,1), np.argmax(preds_total,1), average='macro')
    FINAL_ACC = np.mean(np.argmax(trues_total,1)==np.argmax(preds_total,1))
    print(f'{dir_model} F1-MACRO={FINAL_F1MACRO} ACC={FINAL_ACC}')
    
    # Test preds
    with open(f'{dir_model}preds_test.pkl', 'rb') as file:
        preds_test = pickle.load(file)
        
    valid_preds_list.append(preds_total)
    valid_trues_list.append(trues_total)
    test_preds_list.append(preds_test)


# ## Search the best weighted ensemble

# In[127]:


for i in range(len(best_models_to_ensemble)):
    assert np.sum(valid_trues_list[0]!=valid_trues_list[i])==0


# In[128]:


from scipy.optimize import minimize
from itertools import combinations
combi_iter = combinations(list(np.arange(10)), 3)

    
NUM_MODELS = 9
def func_optim_corr(w):
    w = list(w)
    w.append((len(w)+1.0)-sum(w))
    w = np.array(w)
    w = w / len(w)
    preds_tmp = []
    for numptmp in range(len(COMBI)):
        preds_public = valid_preds_list[COMBI[numptmp]]*w[numptmp]
        preds_tmp.append(preds_public)
    preds_tmp = np.stack(preds_tmp)
    preds_tmp = np.mean(preds_tmp, axis=0)
    score = metrics.f1_score(np.argmax(valid_trues_list[0],1), np.argmax(preds_tmp,1), average='macro')
    
    return -score

for COMBI in combi_iter:
    print(COMBI)
    print(func_optim_corr(np.ones(len(COMBI)-1)))


# In[129]:


# Searching the best combination of models
res = []
for NUM_MODELS in tqdm(range(2,10)): # len(best_models_to_ensemble)+1)):
    combi_iter = combinations(list(np.arange(len(best_models_to_ensemble))), NUM_MODELS)
    for COMBI in combi_iter:
#         print(COMBI)
#         print(func_optim_corr(np.ones(len(COMBI)-1)))
        res_optim = minimize(func_optim_corr, 
                             np.ones(len(COMBI)-1),
                             method='Powell')
        w = list(res_optim.x)
        w.append((len(w)+1.0)-sum(w))
        w = np.array(w)
        w = w / len(w)
#         print('\nNumber of models=', NUM_MODELS, 'F1-MACRO=', -res_optim.fun, '\nW=', w)
        res.append(dict(COMBI=COMBI, SCORE= -res_optim.fun, W=w ))
res = pd.DataFrame(res)
res.head()


# In[130]:


res = res.sort_values('SCORE', ascending=False)
res.to_csv('BestCombi.csv')
res.head()


# In[131]:


res.iloc[0]


# ## Create FINAL submission

# In[132]:


COMBI = res.iloc[0]['COMBI']
res_optim = minimize(func_optim_corr, 
                             np.ones(len(COMBI)-1),
                             method='Powell')
w = list(res_optim.x)
w.append((len(w)+1.0)-sum(w))
w = np.array(w)
w = w / len(w)
print(-res_optim.fun)
print('W=', w)
print(res.iloc[0]['W'])


# In[133]:


preds_test = []
for numptmp in range(len(COMBI)):
    preds_public = test_preds_list[COMBI[numptmp]]*w[numptmp]
    preds_test.append(preds_public)
preds_test = np.stack(preds_test)
preds_test = np.mean(preds_test, axis=0)
print(preds_test[:10,])


# In[134]:


test_df.head()


# In[135]:


# Create Submission
test_df = pd.read_csv('test.csv')
submission=test_df.copy()[['example_path']]
submission['target']= submission['example_path'].str.split('/').str[-1].str.strip('.png')
submission['label'] = np.argmax(preds_test, 1)
submission = submission.drop(columns=['example_path'])
submission = submission.astype({'label':'int','target':'int'})
submission=submission.sort_values(by=['target'])
print(submission.head())
submission.to_csv('predictions.csv',index=False)
submission.set_index('target').rename(columns={"label": "target"}).to_json('predictions.json',orient="columns")


# In[136]:


pd.read_csv('predictions.csv').head()


# In[137]:


get_ipython().system('head predictions.json')


# In[140]:


pd.DataFrame(preds_test).to_csv('preds_test_ensemble_0.8237866251349666.csv', index=False)


# In[ ]:




