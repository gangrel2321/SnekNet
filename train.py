import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import PIL 
from PIL import Image
from collections import OrderedDict
import copy

from SnekData import SnekData
from SnekNet import SnekNet

import random
import argparse
import pandas as pd

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args


seed = 1
torch.manual_seed(seed)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(f"Training on: {device}")

def main(args):
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomRotation((-180,180)), 
            transforms.RandomResizedCrop(240),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid' : transforms.Compose([
            transforms.Resize(240),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {
        'train' : SnekData("TRAIN_SnakeCLEF2022-TrainMetadata.csv", ".", data_transforms['train']),
        'valid' : SnekData("TEST_SnakeCLEF2022-TrainMetadata.csv", ".", data_transforms['valid'])
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                    batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid'] }

    classes = image_datasets['train'].classes
    print("Classes:", classes) 
    train_size = len(image_datasets['train'])
    valid_size = len(image_datasets['valid'])
    print("Training Size:", train_size)
    print("Valid Size:", valid_size)

    # temporary
    model = models.vgg19(pretrained=True) # models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # remove last layer
    features.extend([nn.Linear(num_features, len(classes))]) # create new layer
    model.classifier = nn.Sequential(*features) # add our updates

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=25)

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for datum in dataloaders[phase]:
                inputs = datum['image'].to(device)
                labels = datum['class_id'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    args = create_argparser()
    main(args)