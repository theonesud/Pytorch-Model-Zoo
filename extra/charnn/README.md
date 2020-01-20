# Watson - Char RNN bot

This project implements a minimal Recurrent Neural Network (LSTM) in Tensorflow. The network trains and samples on a character level, in this case from all the Sherlock Holmes novels.

The checkpoints trained on a GTX 1080 for 100 epochs can be found [here](https://drive.google.com/file/d/0B24n6xHwJ0h0MXR3MXRRQUd2N3M/view?usp=sharing). The loss in the end was 1.1275

Here is something it generated after training for 100 epochs:
```
Holmes showed them how we had no such an end of the stuck to my own
country, and he could not hope with the concealed water who were
distinguished at the time in the house. I was sitinally that they had
seen a fair serven and there were nothing to think. I was the lumour
of his professional country and were standing than to tell him.

"I wanted to see that the man that I was able to discover that a cas
is that this secret as well. It is not so many than any of the
matter and a considerable poor frankly. He is the subject which was
closely."

"I take the morning. It is a supper which you have taken out in your
fair first thing."

"What did you make of your house?"

"I have no day?"

"Yes, I can say, and we are never and his son."

"And you cannot assist that? Who is it?"

"I am glad at our face. I would sure the more important than to the
sound which I have not seen the corner which has been a converse of
the most communication as to the sentence window out why we can. The old
cruel to the professor was in my companions and a letter, but the
tamb of the hall was one in the side of the door. A compellette as I
saw, as you see, a dead man without a moment when the clumber of
carriage stepped his constetce. The stairs were not taken about it
with the string. Here he walked out of my professional, but when we at
him as I had always seem to be a man, sir. He had not hit heed against the
passages and had, so I had seen the pleasure of a sergeant that it was
not once to be a lately and table for a muscular. Hunder they were
sure of himself thickly was impendented in this man with him. He came
to the door, and then to send all the room. It came to the clothes of
the most, and which I had confessing that I have been claim in a
delate. I was not far to see that he was always anything, so I would
be supprised."
```

## Requirements

* Python 3.6
* Tensorflow 1.0 (GPU version recommended for training)
* Numpy, Pickle

## Usage

To test the trained network, download [this checkpoints folder](https://drive.google.com/file/d/0B24n6xHwJ0h0MXR3MXRRQUd2N3M/view?usp=sharing) and keep it in the same folder as the code and run:
```
python test.py
```

To train the network use:
```
python train.py
```

To train it on your own dataset link it in *line 45 of train.py*

---
*khatam*
