
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

class SnekNet(nn.Module):
   def __init__(self, vgg: nn.Module, num_classes: int = 1572, init_weights: bool = True, dropout: float = 0.5) -> None:
      super(SnekNet, self).__init__() 
      #vgg = models.vgg19(pretrained=True) # models.resnet18(pretrained=True)
      self.features = vgg.features
      self.avgpool = vgg.avgpool
      self.image_classifier = nn.Sequential(
         nn.Linear(512 * 7 * 7, 512),
         nn.ReLU(True),
         nn.Dropout(p=dropout),
         nn.Linear(512, 512),
      )
      self.meta_classifier = nn.Sequential(
         nn.Linear(3,128),
         nn.ReLU(True),
         nn.Linear(128,128),
      )
      self.final_classifier = nn.Sequential(
         nn.Linear(512 + 128, 2048),
         nn.ReLU(True),
         nn.Dropout(p=dropout),
         nn.Linear(2048, num_classes)
      )

   def forward(self, x, y):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x,1)
      x = self.image_classifier(x)
      y = self.meta_classifier(y)
      z = torch.cat((x.view(x.size(0), -1),
                     y.view(y.size(0), -1)), dim=1)
      z = self.final_classifier(z)
      return z