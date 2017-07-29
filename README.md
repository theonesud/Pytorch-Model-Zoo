# Deep learning models in PyTorch [IN PROGRESS]

This repository contains deep learning models built in [PyTorch](http://pytorch.org/). PyTorch is great for research and intuitive for people new to this field.

The structure for each project is consistent:
* model.py - Contains the model of the neural network.
* train.py - Preprocessing the data and training the network.
* test.py - Infering from the trained network.
* saves folder - Trained models and other saved data can be saved here.
* any other supporting files.

Best practices like dropout, batch normalization, suitable optimizers are used.

## Models

### 1.CNNs
Image Classification: [ResNet 2015](https://arxiv.org/pdf/1512.03385.pdf)

### 2.RNNs
Speech Recognition: [Deep Speech 2 2015](https://arxiv.org/pdf/1512.02595.pdf)
Generating sequences: [LSTM 2013](https://arxiv.org/pdf/1308.0850.pdf)
Sequence to sequence: [Google 2014](https://arxiv.org/pdf/1409.3215.pdf)
Neural Machine Translation: [Bengio 2014](https://arxiv.org/pdf/1409.0473v7.pdf)
Conversational Model: [Google 2015](https://arxiv.org/pdf/1506.05869.pdf)

### 3.GANs
Deep Convolutional GAN: [DCGAN 2015](https://arxiv.org/pdf/1511.06434.pdf)

### 4.Reinforcement

### 5.Others
Hybrid Computing with a NN and external memory [Nature 2016](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf)
