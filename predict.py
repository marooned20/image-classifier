import argparse
import json
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
name = 'PREDICT'
logging.basicConfig(stream=sys.stdout, 
                    level=logging.DEBUG,
                    format='%(name)s - %(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(name)

# constants
VALID_MODELS = {'vgg13': models.vgg13, 
                'vgg16': models.vgg16, 
                'vgg19': models.vgg19}

def create_parser():

    parser = argparse.ArgumentParser(description='Train your neural network')
    parser.add_argument('img_path', type=str,
                        help='path to image including image name')
    parser.add_argument('checkpoint', type=str, 
                        help='path of checkpoint file with file name')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k probabilities')
    parser.add_argument('--category_names', type=str,
                        help='json file path containing mapping of flower names to classes')
    parser.add_argument('--gpu', action='store_true',
                        help='use gpu / CUDA if available')

    return parser


def load_checkpoint(checkpoint, device):
    try:
        if os.path.isfile(checkpoint):
            # exctract model name from file name
            model_name = checkpoint.split('_')[0]

            # load file and model
            checkpoint_meta = torch.load(checkpoint)
            model = VALID_MODELS[model_name](pretrained=True)
            
            model.classifier = checkpoint_meta['classifier']
            model.load_state_dict(checkpoint_meta['state_dict'])
            model.class_to_idx = checkpoint_meta['mapping']
            model.to(device)
            logger.info('running on device {}'.format(device))

            for param in model.parameters():
                # freeze params
                param.requires_grad = False
            logger.info('Freezing parameters')
    except Exception as e:
        logger.error(e)
           
    return model


def process_image(fname):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(fname)
    transformation_params = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    torch_tensor = transformation_params(pil_img)
    # below is an assert for making sure that tensor is created as expected
    assert torch_tensor.numpy().shape == (3, 224, 224), "Dimensions do not match"
    
    return torch_tensor


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    my_tensor1 = process_image(image_path)
    # unsqueeze is required for proper probability extraction
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    my_tensor1.unsqueeze_(0)
    
    model.eval()
    model.to(device)
    
    # my_tensor.to('cuda') does not work
    # solution found on https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html
    my_tensor = my_tensor1.to(torch.device("cuda"))
    logger.info('running on device {}'.format(my_tensor.device))
    
    with torch.no_grad():
        out = model.forward(my_tensor)
    ps = torch.exp(out)
    
    logger.info('Predicting top {} probabilities and classes'.format(topk))
    topk_prob, topk_class = ps.topk(topk)
    
    return topk_prob, topk_class
