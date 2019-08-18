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

def create_parser():

    parser = argparse.ArgumentParser(description='Train your neural network')
    parser.add_argument('data_dir', type=str,
                        help='path to folder where train, valid and test images are stored')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),
                        help='path to folder where trained model will be saved as <arch>_checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='pretrained models from torch.models - choose from "vgg13", "vgg16" or "vgg19"')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='your favourite learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='number of features in the hidden layer')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='How many iterations')
    parser.add_argument('--gpu', action='store_true',
                        help='use gpu / CUDA if available')

    return parser


def transform_data(data_dir, train=True):
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
        # data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else:
        add_transform = [transforms.Resize(256), transforms.CenterCrop(224)]
        out_transforms = transforms.Compose(list(chain(add_transform, base_transform)))
        data = datasets.ImageFolder(data_dir, transform=out_transforms)
        # data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    return data


def data_loader(data):
    return torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)


def load_model(architecture):
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
    logger.debug('params frozen')
    
    return model


def create_classifier(model, hidden_units, output_units=102):

    # make sure hidden units are greater than output_units
    assert hidden_units > output_units, 'Hidden Layer must be greater than 102'
    
    # extract base input features from original (vgg16) model
    input_features = model.classifier[0].in_features

    # define classifier with 3 layers
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_features, 4096)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(p=0.3)),
                                        ('fc2', nn.Linear(4096, hidden_units)),
                                        ('relu2', nn.ReLU()),
                                        ('dropout2', nn.Dropout(p=0.3)),
                                        ('fc3', nn.Linear(512, output_units)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))
    
    model.classifier = classifier


def validate_train_model(model, loader, criterion, device):
    with torch.no_grad():
            
        valid_loss = 0
        accuracy = 0
        model.to(device)
        logger.debug('training model on {}'.format(device))

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            out = model.forward(inputs)
            valid_loss = valid_loss + criterion(out, labels).item()

            ps = torch.exp(out)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy = accuracy + equality.type(torch.FloatTensor).mean()
        
    return valid_loss, accuracy
    

def nn_train(train_loader, valid_loader, model, criterion, optimizer, device, epochs):
    print_every = 30
    for epoch in range(epochs):
        model.to(device)
        model.train()
        for images, labels in train_loader:
            steps = steps + 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # updating only the weights of the feed-forward network
            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            running_loss = running_loss + loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss, accuracy = validate_train_model(model, valid_loader, criterion, device)
                    
                logger.debug("Epoch: {}/{}.. ".format(epoch+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                
                running_loss = 0
                model.train()


def create_checkpoint(model, train_data, save_dir):
    # class to index mapping from training data to model
    model.class_to_idx = train_data.class_to_idx

    # create checkpoint dict with relevant fields
    checkpoint_meta = {'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'mapping': model.class_to_idx,
                'architecture': model.name}

    if not os.path.isdir(save_dir):
        logger.error('Directory path does not exist, please check and try again')

    # save as .pth
    fname = '{}_checkpoint.pth'.format(str(model.name))
    f_dir = os.path.join(save_dir, fname)
    try:
        torch.save(checkpoint_meta, f_dir)
        logger.debug('Saved checkpoint file: {} at {}'.format(fname, save_dir))
    except Exception as e:
        logger.error('Could not save checkpoint file. Error: {}'.format(e))
    # TODO: Add finally block in case of invalid save_dir, to prevent losing model


def main():
    parser = create_parser()

    # parse arguments
    args = parser.parse_args()

    # assign values
    data_dir = args.data_dir
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    arch = args.arch
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    # TODO: remove print statement later
    print(data_dir, learning_rate, save_dir, arch, hidden_units, epochs, gpu, device)

    # transform and load data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = transform_data(train_dir)
    valid_data = transform_data(valid_dir, train=False)
    test_data = transform_data(test_dir, train=False)

    train_loader = data_loader(train_data)
    valid_loader = data_loader(valid_data)
    test_loader = data_loader(test_data)

    # load model
    model = load_model(arch)

    # create classifier
    create_classifier(model, hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # train neural network
    nn_train(train_loader, valid_loader, model, criterion, optimizer, device, epochs)

    # quick test/validation of the network
    test_loss, accuracy = validate_train_model(model, test_loader, criterion, device)
    logger.debug("test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

    # save checkpoint
    create_checkpoint(model, train_data, save_dir)


main()