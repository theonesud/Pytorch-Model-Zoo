from torchvision import models
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19(pretrained=True)

    def forward(self, x):
        features = []
        for name, layer in self.vgg.features.named_children():
            # This will manually pass x through all the layers and save
            # only the five mentioned activations and return them
            # Try print(self.vgg) for more info
            x = layer(x)
            if name in ['0', '5', '10', '19', '28']:
                features.append(x)
        return features
