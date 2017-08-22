import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# CIFAR-10 Dataset
test_dataset = dsets.CIFAR10(root='../datasets/',
                             train=False,
                             transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# Model
resnet = torch.load('../trained/resnet.pkl')

# Test
correct = 0
total = 0
for images, labels in test_loader:
    images = images.cuda()
    labels = labels.cuda()
    images = Variable(images)
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the test images: %d %%' %
      (100 * correct / total))
