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
from sklearn.metrics import f1_score as f1_score_metric

from SnekData import SnekData
from SnekNet import SnekNet

import random
import argparse
import pandas as pd
import time
from tqdm import tqdm
import os

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    return args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

seed = 1
torch.manual_seed(seed)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Training on: {device}")

def main(args):
    data_transforms = {
        'train' : transforms.Compose([
            #transforms.RandomRotation((-180,180)), 
            transforms.RandomResizedCrop((240,240)),
            transforms.ToTensor(),
            #transforms.Resize((240,240)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid' : transforms.Compose([
            transforms.Resize((240,240)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {
        'train' : SnekData("training.csv", args.data_dir, data_transforms['train']),
        'valid' : SnekData("validation.csv", args.data_dir, data_transforms['valid'])
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                    batch_size=args.batch_size, shuffle=True, num_workers=32) for x in ['train', 'valid'] }

    classes = image_datasets['train'].classes
    print("Number of Classes:", len(classes))
    print("Classes:", classes) 
    train_size = len(image_datasets['train'])
    valid_size = len(image_datasets['valid'])
    dataset_sizes = {
        'train' : train_size,
        'valid' : valid_size
    }
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

    optimizer_ft = optim.Adam(model.parameters(), lr=0.005, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-07) #optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = nn.DataParallel(model)

    model = model.to(device)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()
    scaler = torch.cuda.amp.GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            f1_score = 0

            # Iterate over data.
            for datum in tqdm(dataloaders[phase]):
                inputs = datum['image'].to(device)
                labels = datum['label_id'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        outputs = torch.nn.functional.softmax(outputs,-1)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs,dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                f1_score += f1_score_metric(labels.cpu().data, preds.cpu(), average='macro')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            mean_f1 = f1_score / (dataset_sizes[phase] / args.batch_size)

            print(f"Mean F1-Score: {mean_f1}")
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                print("Saving...")
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"{args.log_dir}/{epoch}_best_weights.pt")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

if __name__ == "__main__":
    args = create_argparser()
    main(args)