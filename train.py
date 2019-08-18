import argparse
import logging
import os
import random
import sys
from collections import OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision import datasets, models, transforms

# configure logger
name = 'TRAIN'
logging.basicConfig(stream=sys.stdout, 
                    level=logging.DEBUG,
                    format='%(name)s - %(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(name)

# constants
VALID_MODELS = ['vgg13', 'vgg16', 'vgg19']

def transform_load(data_dir, train=True):
    # check if directory actually exists
    if not os.path.isdir(data_dir):
        logger.error('Directory path does not exist, please check and try again')

    base_transform = [transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]

    if train:
        add_transform = [transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip()]
        out_transforms = transforms.Compose(list(chain(add_transform, base_transform)))
        data = datasets.ImageFolder(data_dir, transform=out_transforms)
        data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else:
        add_transform = [transforms.Resize(256), transforms.CenterCrop(224)]
        out_transforms = transforms.Compose(list(chain(add_transform, base_transform)))
        data = datasets.ImageFolder(data_dir, transform=out_transforms)
        data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    return data_loader


def load_model(architecture='vgg16'):
    if architecture in VALID_MODELS:
        try:
            exec("model = models.{}(pretrained=True)".format(architecture))
            model.name = architecture
        except Exception as e:
            logger.error(e)
    else:
        logger.error('Please select model from {}'.format(str(VALID_MODELS)))

    # freeze to avoid backpropogation
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def create_classifier():
    
    pass