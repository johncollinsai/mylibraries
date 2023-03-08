import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing


def initialize_random_state(random_seed):
    """Initialize the random states of all relevant modules, such that tensorflow model results are reproducible.

    Args:
        random_seed (int): random seed

    Return:
        None

    """
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    import random
    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    import tensorflow as tf
    tf.random.set_seed(random_seed)

    from tensorflow.compat.v1.keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)






class PerSampleLossHistory(keras.callbacks.Callback):
    # The logs.get('loss') number is the loss value averaging all the previously evaluated batches.
    # If we want to get the loss solely on that batch, we need to rely on the complex formula below
    # Let mu_k be the averaged loss at index k
    # loss_k = (k + 1) mu_k - k * mu_(k-1)
    # self.prev_avg keeps track of mu_(k-1), self.idx keeps track of k
    def __init__(self, logs={}):
        self.losses = []
        self.prev_avg = 0
        self.idx = 0

    def on_test_begin(self, logs={}):
        self.idx = 0
        self.losses = []

    def on_test_batch_end(self, batch, logs={}):
        if self.idx == 0:
            self.losses.append(logs.get('loss'))
        else:
            value = ((self.idx + 1) * logs.get('loss')) - (self.idx * self.prev_avg)
            self.losses.append(value)
        self.prev_avg = logs.get('loss')
        self.idx = self.idx + 1

if __name__ == '__main__':
    pass
