import torch
from torch.autograd import Variable
import torchvision

sample_size = 100
z_dim = 100


def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Model
g = torch.load('../trained/generator.pkl')
d = torch.load('../trained/discriminator.pkl')

# Set them to evaluation mode
g.eval()
d.eval()

# Test
noise = Variable(torch.randn(sample_size, z_dim))
fake_images = g(noise)
torchvision.utils.save_image(denorm(fake_images.data), 'sampled.png', nrow=12)
