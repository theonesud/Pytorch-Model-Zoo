# Deep learning models in PyTorch [IN PROGRESS]

This repository contains deep learning models built in [PyTorch](http://pytorch.org/). Intended for learning PyTorch this repo is made very simple to understand for a person with basic python and deep learning knowledge. All kinds of important deep learning models are implemented here. Links to the papers are also given.

The structure for each project is consistent:
* model.py - Contains the model of the neural network.
* train.py - Preprocessing the data and training the network.
* test.py - Infering from the trained network.
* any other supporting files.

Common folders:
Trained folder - Trained models can will be saved here.
Datasets folder - Datasets downloaded will be stored here.

Best practices like PEP8, dropout, batch normalization, suitable optimizers are used.

My trained models can be found [here](https://drive.google.com/open?id=0B24n6xHwJ0h0TW5mdWk2QTZIN0k).

### IMP: For training on the CPU remove '.cuda()' wherever you find it.

## Models

### 1. CNNs
* Image Classification: [Microsoft-ResNet 2015](https://arxiv.org/pdf/1512.03385.pdf)
* Spatial Transformation Network [STN 2016](https://arxiv.org/pdf/1506.02025.pdf)
* You Only Look Once [YOLO 2016](https://arxiv.org/pdf/1506.02640.pdf)
* Super resolution [IEEE 2015](https://arxiv.org/pdf/1501.00092v3.pdf)
* Artistic Style Transfer [Gatys 2015](https://arxiv.org/pdf/1508.06576.pdf)
* Deep Photo Style Transfer [2017](https://arxiv.org/pdf/1703.07511v1.pdf)
* Neural Doodle [2016](https://arxiv.org/pdf/1603.01768.pdf)
* Image Colorization [UC Berkeley 2016](https://arxiv.org/pdf/1603.08511.pdf)

### 2. RNNs
* Speech Recognition: [Deep Speech 2 2015](https://arxiv.org/pdf/1512.02595.pdf) (If possible with [attention](https://arxiv.org/pdf/1508.04395.pdf)) (Maybe [Deep Voice](https://arxiv.org/pdf/1702.07825v2.pdf) )
* Generating sequences: [LSTM 2013](https://arxiv.org/pdf/1308.0850.pdf)
* Sequence to sequence with attention: [Text Summarizer](https://github.com/tensorflow/models/tree/master/textsum)
* Neural Machine Translation: [Google 2016](https://arxiv.org/pdf/1609.08144.pdf)
* Conversational Model: [Google 2015](https://arxiv.org/pdf/1506.05869.pdf)
* Skip Thoughts [sentence to vector](https://arxiv.org/pdf/1506.06726.pdf)

### 3. GANs
* Deep Convolutional GAN: [DCGAN 2015](https://arxiv.org/pdf/1511.06434.pdf)
* Text to image synthesis [2016](https://arxiv.org/pdf/1605.05396v2.pdf)

### 4. Reinforcement
* Deep Q Learning [Deep Mind 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* Asynchronous advantage actor-critic [A3C](https://arxiv.org/pdf/1602.01783.pdf)

### 5. Others
* Image Captioning with attention [Bengio 2016](https://arxiv.org/pdf/1502.03044.pdf)
* Hybrid Computing with a NN and external memory [Nature 2016](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf)
* Network in Network [2014](https://arxiv.org/pdf/1312.4400.pdf)

References:
[https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)
[https://github.com/terryum/awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers)
[https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
[https://github.com/pytorch/examples](https://github.com/pytorch/examples)
[https://github.com/bharathgs/Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list)
[https://github.com/ritchieng/the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)
