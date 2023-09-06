import torch
from model import RGVOS

model = RGVOS()

torch.save(model.state_dict(), './trained-vos-imagenet.pt')
