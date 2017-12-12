import torchvision.datasets as dsets
from torchvision import transforms
from collections import Counter
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from model import EncoderCNN, DecoderRNN
import os.path


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        print('###', cap)
        end = lengths[i]
        targets[i, :end] = cap[0][:end]
    return images, targets, lengths


transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])

train_dataset = dsets.CocoCaptions(root='../datasets/train2014',
                                   annFile='../datasets/annotations/captions_train2014.json',
                                   transform=transform)

if not os.path.isfile('img_cap_vocab.pkl'):
    counter = Counter()
    for img, captions in train_dataset:
        for caption in captions:
            counter.update(caption.rstrip('.').lower().split())
    words = set([word for word, cnt in counter.items() if cnt >= 4])
    words.add('<pad>')
    words.add('<start>')
    words.add('<end>')
    words.add('<unk>')

    word_to_int = {w: i for i, w in enumerate(words)}
    int_to_word = dict(enumerate(words))
    with open('img_cap_vocab.pkl', 'wb+') as f:
        pickle.dump((word_to_int, int_to_word), f)

with open('img_cap_vocab.pkl', 'rb') as f:
    word_to_int, int_to_word = pickle.load(f)


data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last=True,
                                          collate_fn=collate_fn)


for images, captions in data_loader:
    print(images, captions)









# encoder = EncoderCNN(256)
# decoder = DecoderRNN(256, 512, len(int_to_word), 1)

# encoder.cuda()
# decoder.cuda()










# criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam([{'params': decoder.parameters()},
#                               {'params': encoder.parameters()}], lr=0.001)

# num_epochs = 1

# total_step = len(train_dataset)
# for epoch in range(num_epochs):
#     for i, (images, captions) in enumerate(train_dataset):

#         # Set mini-batch dataset
#         images = Variable(images.cuda(), volatile=True)
#         captions = captions[0]
#         targets = torch.nn.utils.rnn.pack_padded_sequence(captions, batch_first=True)[0]

#         # Forward, Backward and Optimize
#         decoder.zero_grad()
#         encoder.zero_grad()
#         features = encoder(images)
#         outputs = decoder(features, captions)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         # Print log info
#         if i % 100 == 0:
#             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
#                   % (epoch, num_epochs, i, total_step,
#                      loss.data[0], np.exp(loss.data[0])))

#         # Save the models
#         if (i + 1) % 100 == 0:
#             torch.save(decoder.state_dict(), 'decoder.pkl')
#             torch.save(encoder.state_dict(), 'encoder.pkl')
