import numpy as np
import time
import tensorflow as tf
import pickle
from model import CharRNN
import os


epochs = 100
save_every_n = 1000


def delete_checkpoints():
    """
    Deletes all the contents in the checkpoints folder
    """
    folder = './saves/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_batches(arr, n_seqs, n_steps):
    """
    Generator function that yields batches
    """
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr) // characters_per_batch
    arr = arr[:n_batches * characters_per_batch]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def train():

    # Preprocessing
    with open('holmes.txt', 'r') as f:
        text = f.read()
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    no_classes = len(vocab)
    pickle.dump((int_to_vocab, vocab_to_int, no_classes),
                open('./saves/data.p', 'wb'))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    # Initialize the model
    model = CharRNN(no_classes=no_classes)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0

        # Training
        for e in range(epochs):
            new_state = sess.run(model.initial_state)
            for x, y in get_batches(encoded, model.no_seqs, model.seq_len):
                counter += 1
                start = time.time()
                feed = {model.inputs: x, model.targets: y,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run(
                    [model.loss, model.final_state, model.train_op], feed_dict=feed)
                end = time.time()
                print('Epoch: {} '.format(e + 1), 'Loss: {:.4f} '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

                if (counter % save_every_n == 0):
                    saver.save(sess, "saves/{}.ckpt".format(counter))

        saver.save(sess, "saves/{}.ckpt".format(counter))


if __name__ == '__main__':
    if not os.path.isdir('./saves'):
        os.mkdir('./saves')
    delete_checkpoints()
    train()
