cd img\ captioning/pycoco/

pycocotools v2.0

git submodule init
git submodule update

cd PythonAPI
make
python setup.py build
python setup.py install

Download the [COCO training images 2014 dataset](http://images.cocodataset.org/zips/train2014.zip) into datasets
and the [annotations 2014](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

The directory structure will look like:
```
|__ datasets
    |__ train2014
        |__ img1.jpg
        |__ img2.jpg
        |__ img3.jpg
        |__ img4.jpg
        |__ ...
    |__ annotations
        |__ annotaion1.json
        |__ annotaion2.json
        |__ annotaion3.json
        |__ ...
```
