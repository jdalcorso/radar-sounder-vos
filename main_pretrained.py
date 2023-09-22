import torch
import torch.nn as nn
from model import RGVOS

model = RGVOS()

model = nn.DataParallel(model)

torch.save(model.state_dict(), './trained-vos-imagenet.pt')
print('saved pretrained resnet18')