import torch
from torchvision import transforms, datasets, models
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    #model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
            param.requires_grad = False
            
    # create classifier
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(checkpoint['n_in'], checkpoint['n_h'][0])),   
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(checkpoint['n_h'][0],checkpoint['n_h'][1])),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.5)),
                            ('fc3', nn.Linear(checkpoint['n_h'][1], checkpoint['n_out'])),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    # put classifier on pretrained network
    model.classifier = classifier
    
    
    model.optimizer = checkpoint['optimizer_state.dict']
    
    model.class_to_idx = checkpoint['class_to_idx:']

    model.load_state_dict(checkpoint['state_dict'])
    
    return model




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # get image and resize either height or width to 256 depending on which is shorter
    image = Image.open(image)
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    
    #crop the center to 224 x 224
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    image = image.crop((left_margin, bottom_margin, right_margin,    
                   top_margin))
    
    image = np.array(image)/255
    
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image


def predict(image, model, device='cpu', top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
  
    model.to(device)
    
    # TODO: Implement the code to predict the class from an image file
    result_classes = []
    result_index = []
    c_i = {}
    
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    log_results = model.forward(image)
    pss = torch.exp(log_results)
    top_probs, top_indices = pss.topk(5)
    top_indices = top_indices.data.numpy().tolist()[0]
    
    # convert top_classes and probabilies to list
   
    top_probs = top_probs.data.numpy().tolist()[0]


    # loop through class to indices: v is indices, m is classes
    for m,v in model.class_to_idx.items():
         for i in top_indices:
                if v == int(i):
                    #type(m)
                    result_classes.append(m)
                
    
    
    return top_probs, result_classes



def find_flower_names(classes, categories):

    flowers = []
    for index, name in categories.items():
        for i in classes:
            if index == i:
                flowers.append(name)
                   
    return flowers


