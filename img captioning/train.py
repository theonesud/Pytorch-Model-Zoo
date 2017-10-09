# import torchvision.datasets as dsets

# train_dataset = dsets.CocoCaptions(root='../datasets/train2014',
#                                    annFile='../datasets/annotations/captions_train2014.json')
from pycocotools.coco import COCO

coco = COCO('../datasets/annotations/captions_train2014.json')
imgids = coco.getImgIds()
print('imgids: ', len(imgids))
anns = []
keyerrors = []
for img in imgids:
    try:
        anns.append(coco.loadAnns(img))
    except KeyError:
        keyerrors.append(img)

print('anns: ', len(anns))
print('errors: ', len(keyerrors))

imgids = list(coco.anns.keys())
print('anns keys', len(imgids))
