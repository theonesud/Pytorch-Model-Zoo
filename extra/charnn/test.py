import tensorflow as tf
import pickle
import numpy as np
from model import CharRNN


n_samples = 5000
prime = '     Holmes '


def pick_top_n(preds, vocab_size):
    """
    Picks a prediction from the top 5 outputs from softmax
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-5]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint):
    samples = [c for c in prime]
    int_to_vocab, vocab_to_int, no_classes = pickle.load(
        open("./saves/data.p", "rb"))

    # Initialize the model
    model = CharRNN(no_classes=no_classes, sampling=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Load the checkpoint
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)

        # Feed the prime word to the model and predict the next character
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x, model.initial_state: new_state}
            preds, new_state = sess.run(
                [model.prediction, model.final_state], feed_dict=feed)
        c = pick_top_n(preds, no_classes)
        samples.append(int_to_vocab[c])

        # Generate new samples
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x, model.initial_state: new_state}
            preds, new_state = sess.run(
                [model.prediction, model.final_state], feed_dict=feed)
            c = pick_top_n(preds, no_classes)
            samples.append(int_to_vocab[c])

    return ''.join(samples)


if __name__ == '__main__':

    checkpoint = tf.train.latest_checkpoint('saves')
    samp = sample(checkpoint)
    print(samp)
