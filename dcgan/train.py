import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from model import Generator, Discriminator

# Hyperparameters
lr = 0.0002
batch_size = 32
image_size = 64
z_dim = 100
epochs = 80

# Image Preprocessing
transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

# LFW DeepFunneled Dataset
dataset = ImageFolder(root='../datasets/lfw-deepfunneled/',
                      transform=transform)

# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last=True)

# Model
g = Generator(z_dim=z_dim,
              image_size=image_size,
              conv_dim=64)
d = Discriminator(image_size=image_size,
                  conv_dim=64)
d = d.cuda()
g = g.cuda()

# Optimizers and loss
g_optimizer = torch.optim.Adam(g.parameters(),
                               lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d.parameters(),
                               lr, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

# Training
for epoch in range(epochs):
    for i, (images, _) in enumerate(data_loader):

        zeros = Variable(torch.zeros(batch_size).cuda())
        ones = Variable(torch.ones(batch_size).cuda())

        # Passing through generator
        noise = Variable(torch.randn(batch_size, z_dim).cuda())
        fake_images = g(noise)
        outputs = d(fake_images)
        g_loss = criterion(outputs, ones)

        d.zero_grad()
        g.zero_grad()

        g_loss.backward()
        g_optimizer.step()

        # Passing through discriminator
        images = Variable(images.cuda())
        outputs = d(images)
        real_loss = criterion(outputs, ones)

        noise = Variable(torch.randn(batch_size, z_dim).cuda())
        fake_images = g(noise)
        outputs = d(fake_images)
        fake_loss = criterion(outputs, zeros)

        d_loss = real_loss + fake_loss

        d.zero_grad()
        g.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % batch_size == 0:
            print('Epoch [%d/%d], Iter [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f' %
                  (epoch + 1, epochs, i + 1, len(data_loader),
                   real_loss.data[0], fake_loss.data[0],
                   g_loss.data[0]))

# Save the Model
torch.save(g, '../trained/generator.pkl')
torch.save(d, '../trained/discriminator.pkl')
