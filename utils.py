import numpy as np
import tensorflow as tf

def ps(a0):
    print(np.shape(a0))

def tp(t):
    return tf.Print([0], [t])
