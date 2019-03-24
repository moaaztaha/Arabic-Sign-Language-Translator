##################################################
# Imports 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, copy
#####################################################
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
class_names = ['ain',
 'al',
 'aleff',
 'bb',
 'dal',
 'dha',
 'dhad',
 'fa',
 'gaaf',
 'ghain',
 'ha',
 'haa',
 'jeem',
 'kaaf',
 'khaa',
 'la',
 'laam',
 'meem',
 'nun',
 'ra',
 'saad',
 'seen',
 'sheen',
 'ta',
 'taa',
 'thaa',
 'thal',
 'toot',
 'waw',
 'ya',
 'yaa',
 'zay']
# Load the pretrianed model
model = models.vgg16(pretrained=True) 

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
    
# Add new classifier
model.classifier[6] = nn.Sequential(
                        nn.Linear(in_features=4096, out_features=256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(in_features=256, out_features=32),
                        nn.LogSoftmax(dim=1)
                    )

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# send the model to gpu
model.to(device)




def reLoadCheckpoint(model, optimizer, path, train=True):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if train == False:
        model.eval()
    else:
        model.train()
    return model

model = reLoadCheckpoint(model, optimizer, 'ArASL_Database_54K_Final/model/checkpoint.pth.tar', train=False)

def process_image(image):
    """Process an image path into a PyTorch tensor"""

    image = Image.fromarray(image)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    img = rgbimg
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    return img

def classify(img):
    with torch.no_grad():
        model.eval()
        image = process_image(img)
        image = torch.Tensor(image)
        image = image.view(1, 3, 224, 224)
        out = model(image.to(device)) # removec .cuda
        _, ps = torch.max(out,1)    
        return class_names[ps]