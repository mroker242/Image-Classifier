import numpy as np

from torchvision import transforms, datasets, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import helperTrain as helper
import argparse

# set up argument parser. Set up directory argument option
parser = argparse.ArgumentParser(description = 'Image Classifier App Arguments')
parser.add_argument('directory')
parser.add_argument('-s', '--save_dir')
parser.add_argument('-a', '--arch')
parser.add_argument('-l', '--learning_rate')
parser.add_argument('-g', '--gpu')
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-H', '--hidden_units')
args = parser.parse_args()



# initialize some variables
learning_rate = 0.001
epochs = 7
arch = 'vgg16'
hidden_units = [4096, 800]
gpu = 'cuda'

if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.gpu:
    gpu = args.gpu
    


# load data

data_dir = args.directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print('in progress.../')
# All transforms
data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.CenterCrop(224), # this is to reduce the noise from the data
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])])

# Validation transform
data_valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])])
# Test transform
data_test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)

image_testdata = datasets.ImageFolder(test_dir, transform=data_test_transforms)

image_validdata = datasets.ImageFolder(train_dir, transform=data_valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
dataloaders_test = torch.utils.data.DataLoader(image_testdata, batch_size=64)
dataloaders_valid = torch.utils.data.DataLoader(image_validdata, batch_size=64)

# label mapping
with open('../aipnd-project/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# build initial model  
model = helper.model_call(arch)

optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# Defining the size of our network, input, output, layers
n_in, n_h, n_out = 25088, hidden_units, 102



# BUILD THE MODEL
helper.load_model(model, nn, OrderedDict, n_in, n_h, n_out)

# train the model with parameters model, learning rate, epochs
helper.train_model(model, 0.001, epochs, nn, optim, dataloaders, validation_data=dataloaders_valid)

# save the model using argument given, if not use testing.pth as default
if args.save_dir:
    helper.save_model(args.save_dir, model, optimizer, image_datasets, arch)
else:
    helper.save_model('testing.pth', model, optimizer, image_datasets, arch)
        
    








