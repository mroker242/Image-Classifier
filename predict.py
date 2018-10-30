import numpy as np
from torchvision import transforms, datasets, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import helperPredict as helper
import argparse
import json

parser = argparse.ArgumentParser(description = 'Image Classifier App Arguments')
parser.add_argument('path_to_image')
parser.add_argument('path_to_checkpoint')
parser.add_argument('-k', '--top_k')
parser.add_argument('-j', '--add_json')
parser.add_argument('-g', '--gpu')
args = parser.parse_args()

if args.add_json:
    add_json = args.json
else:
    add_json = '../aipnd-project/cat_to_name.json'
    
if args.gpu:
    device = args.gpu
else:
    device = 'cpu'
    
if args.top_k:
    top_k = args.top_k
else:
    top_k = 5
    
with open('../aipnd-project/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



if args.top_k:
    top_k = args.top_k
else:
    top_k = 5

# Load default model 'testing.pth' otherwise load model provided
if args.path_to_checkpoint:
    model = helper.load_checkpoint(args.path_to_checkpoint)
else:
    model = helper.load_checkpoint('testing.pth')


#print(helper.predict(args.path_to_image, model, top_k))

image_path = '../aipnd-project/flowers/test/102/image_08042.jpg'

pss, cls = helper.predict(args.path_to_image, model, device, top_k) # get probabilities and classes
print('Probabilities: {} \nClasses: {}'.format(pss,cls))
flwrs = helper.find_flower_names(cls, cat_to_name)

print('predicted Flower: ',flwrs[0])







