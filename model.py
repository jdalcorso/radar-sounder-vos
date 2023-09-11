import torch
import torch.nn as nn
import torchvision.models.resnet as torch_resnet
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock


class RGVOS(nn.Module):
    def __init__(self, encoder = None):
        super().__init__()
        self.encoder_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        self.encoder = torch_resnet.ResNet(block = BasicBlock, layers = [2, 2, 2, 2])
        self.fc0 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = (1,1))
        state_dict = load_state_dict_from_url(self.encoder_url, model_dir = './radar_vos')
        self.encoder.load_state_dict(state_dict)

        encoding_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*encoding_layers)

        # Modify stride to (1, 1) everywhere in encoder layer
        # When modifying padding, one should also modify conv2 padding
        for layer in self.encoder.children():
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, BasicBlock):
                        sublayer.conv1.stride = (1,1)
                        sublayer.conv1.padding_mode = 'replicate'
                        sublayer.conv2.padding_mode = 'replicate'
                        if isinstance(sublayer.downsample, nn.Sequential):
                            sublayer.downsample[0].stride = (1,1)
                            sublayer.downsample[0].padding_mode = 'replicate'

        #print(list(self.encoder.children()))

    def forward(self,v):

        # fc to get 3 channels
        #x = self.fc0(v[:,:,0,:,:])
        #y = self.fc0(v[:,:,1,:,:])

        # repeat to get 3 channels
        x = v[:,:,0,:,:]
        y = v[:,:,1,:,:]

        x = self.encoder(x)
        y = self.encoder(y)
        return x,y