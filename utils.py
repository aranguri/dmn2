import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def ps(a0):
    print(np.shape(a0))

def psf(t):
    return tf.matmul(t, tf.constant(0.0))

def psi(t):
    return tf.matmul(t, tf.to_int32(tf.constant(0)))

def ps2(t, t2):
    return tf.matmul(t, t2)

def tp(t, summarize=10000):
    return tf.Print([0], [t], summarize=summarize)

def tps(ts, summarize=10000):
    return tf.Print([0], ts, summarize=summarize)

def mm(t):
    return tf.matmul(t, t)

def smooth_plot(values, smooth_factor=10):
    plot([np.mean(list(values.values())[0:(i + 1)]) if i < (smooth_factor + 1) else np.mean(list(values.values())[i-smooth_factor:(i + 1)]) for i in range(len(values))])

def plot(array):
    plt.ion()
    plt.cla()
    if type(array) is dict:
        array = [v for v in array.values()]
    xlim = 2 ** (1 + int(np.log2(len(array))))
    ylim = 2 ** (1 + int(np.log2(np.maximum(max(array), 1e-8))))

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.plot(array)
    plt.pause(1e-8)

def print_all_vars(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    print(values[5][0])
    '''
    for k, v in zip(variables_names, values):
        print ("Variable: ", k)
        print ("Shape: ", v.shape)
        print (v[0][0])
    '''

def print_gradients(sess, loss, feed_dict):
    gradients = tf.gradients(loss, tf.trainable_variables())
    gradients_ = sess.run(gradients, feed_dict)

    for var, grad in zip(tf.trainable_variables(), gradients_):
        if np.shape(grad) == (3,):
            grad = grad[0]

        norm = np.linalg.norm(grad)
        value = sess.run(var, feed_dict)
        print (var.name, norm)
