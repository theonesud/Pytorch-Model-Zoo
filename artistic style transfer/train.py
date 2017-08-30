import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from model import VGGNet

# Hyperparameters
content_image = 'content.jpg'
style_image = 'style.jpg'
max_size = 400
style_weight = 100
epochs = 1000

# Transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])

# Content Image processing
content_image = Image.open(content_image)
scale = max_size / max(content_image.size)
size = np.array(content_image.size) * scale
content_image = content_image.resize(size.astype(int), Image.ANTIALIAS)
content_image = transform(content_image).unsqueeze(0).cuda()

# Style Image processing
style_image = Image.open(style_image)
style_image = style_image.resize([content_image.size(2),
                                  content_image.size(3)], Image.LANCZOS)
style_image = transform(style_image).unsqueeze(0).cuda()

# Initialize result and optimizer
result_image = Variable(content_image.clone(), requires_grad=True)
optimizer = torch.optim.Adam([result_image], lr=0.003, betas=[0.5, 0.999])

# Model
vgg = VGGNet()
vgg = vgg.cuda()

# Train
for step in range(epochs):

    target_features = vgg(result_image)
    content_features = vgg(Variable(content_image))
    style_features = vgg(Variable(style_image))

    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):

        # Content loss
        content_loss += torch.mean((f1 - f2)**2)

        # Reshape conv features
        _, c, h, w = f1.size()
        f1 = f1.view(c, h * w)
        f3 = f3.view(c, h * w)

        # Compute gram matrix
        f1 = torch.mm(f1, f1.t())
        f3 = torch.mm(f3, f3.t())

        # Style loss
        style_loss += torch.mean((f1 - f3)**2) / (c * h * w)

    loss = content_loss + style_weight * style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        print ('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f'
               % (step + 1, epochs, content_loss.data[0], style_loss.data[0]))

    if (step + 1) % 10 == 0:
        # Save the generated image
        denorm = transforms.Normalize(
            (-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = result_image.clone().cpu().squeeze()
        img = denorm(img.data).clamp_(0, 1)
        torchvision.utils.save_image(img, 'output/output-%d.png' % (step + 1))
