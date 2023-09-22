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
        state_dict = load_state_dict_from_url(self.encoder_url, model_dir = './radar_vos')
        self.encoder.load_state_dict(state_dict)

        encoding_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*encoding_layers)

        # Modify stride to (1, 1) everywhere in encoder layer, modify padding
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv2d):
                layer.padding = 'same'
                layer.padding_mode = 'replicate'
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, BasicBlock):
                        sublayer.conv1.stride = (1,1)
                        sublayer.conv1.padding_mode = 'replicate'
                        sublayer.conv2.padding_mode = 'replicate'
                        sublayer.conv1.padding = 'same'
                        sublayer.conv2.padding = 'same'
                        if isinstance(sublayer.downsample, nn.Sequential):
                            sublayer.downsample[0].stride = (1,1)
                            sublayer.downsample[0].padding_mode = 'replicate'
                            sublayer.downsample[0].padding = 'same'

        # Freeze all block except the last one
        # i = 0
        # for layer in self.encoder:
        #     i = i + 1
        #     if i != 8:
        #         for param in layer.parameters():
        #             param.requires_grad = False

        # Print total number of trainable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        # Print specs of all layers of the encoder
        # print(list(self.encoder.children()))

    def forward(self,v):
        if len(v.shape)>4:
            v1 = self.encoder(v[:,:,0,:,:])
            v2 = self.encoder(v[:,:,1,:,:])
            return v1,v2
        
        # Single image encoding
        else:
             x = v[:,:,:,:]
             return self.encoder(x)