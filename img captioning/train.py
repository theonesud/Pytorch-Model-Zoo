import torchvision.datasets as dsets
from torchvision import transforms
from collections import Counter
import pickle
from pycocotools.coco import COCO
from model import EncoderCNN, DecoderRNN

coco_cap = COCO('../datasets/annotations/captions_train2014.json')
ann_ids = coco_cap.anns.keys()

counter = Counter()
for i, id in enumerate(ann_ids):
    caption = str(coco_cap.anns[id]['caption'])
    counter.update(caption.lower().split())
words = set([word for word, cnt in counter.items() if cnt >= 4])
words.add('<pad>')
words.add('<start>')
words.add('<end>')
words.add('<unk>')

word_to_int = {w: i for i, w in enumerate(words)}
int_to_word = dict(enumerate(words))
with open('../trained/img_cap_vocab.pkl', 'wb+') as f:
    pickle.dump((word_to_int, int_to_word), f)

train_dataset = dsets.CocoCaptions(root='../datasets/train2014',
                                   annFile='../datasets/annotations/captions_train2014.json')

transform = transforms.Compose([transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])

encoder = EncoderCNN(256)
decoder = DecoderRNN(256, 512, len(int_to_word), 1)

encoder.cuda()
decoder.cuda()
