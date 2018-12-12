import tensorflow as tf
from preprocess import Babi_task

babi_task = Babi_task(num_task=1, batch_size=32)

input_ = tf.placeholder(tf.float32, shape=(batch_size, input_length, input_size))
question = tf.placeholder(tf.float32, shape=(batch_size, input_length, input_size))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    input_, question, answer, label = babi_task.next_batch()
