import itertools
import tensorflow as tf
from preprocess import BabiTask
from model import DMNCell
import numpy as np
from utils import *

batch_size = 32
embeddings_size = 20

babi_task = BabiTask(batch_size)
input_length, question_length, vocab_size = babi_task.get_lengths()

input_ids = tf.placeholder(tf.int32, shape=(batch_size, input_length))
question_ids = tf.placeholder(tf.int32, shape=(batch_size, question_length))
answer = tf.placeholder(tf.int32, shape=(batch_size, vocab_size))
optimize = tf.placeholder(tf.bool, shape=())

embeddings = tf.random_normal((vocab_size, embeddings_size), stddev=.1)
input = tf.nn.embedding_lookup(embeddings, input_ids)
question = tf.nn.embedding_lookup(embeddings, question_ids)
eos_vector = tf.nn.embedding_lookup(embeddings, babi_task.eos_vector)

dmn_cell = DMNCell(eos_vector, vocab_size, h_size=30, similarity_layer_size=30,
                   num_passes=1, learning_rate=1e-4, reg=1e-3)
loss, accuracy, output = dmn_cell.run(input, question, answer, optimize)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_acc, dev_loss, dev_acc = {}, {}, {}, {}

    for j in itertools.count():
        input_, question_, answer_ = babi_task.next_batch()
        feed_dict = {input_ids: input_, question_ids: question_, answer: answer_, optimize: True}
        tr_loss[j], tr_acc[j], output_ = sess.run([loss, accuracy, output], feed_dict)

        print(f'Loss: {tr_loss[j]}. Accuracy: {tr_acc[j]}. Output: {np.argmax(output_, axis=1)[:15]}')
        smooth_plot(tr_acc)

        if j % 10 == 0:
            input_, question_, answer_ = babi_task.dev_data()
            feed_dict = {input_ids: input_, question_ids: question_, answer: answer_, optimize: False}
            dev_loss[j], dev_acc[j], output_ = sess.run([loss, accuracy, output], feed_dict)
            print(f'** Dev: Loss: {dev_loss[j]}. Accuracy: {dev_acc[j]}. **')

'''
Before it's working
* making the code more concise
* try to understand the code and check whether is correct.
* change eq 8 to a sofmtax
* add eval data (need to modify dmn to allow not optimizing)
* train for a long time
* dropout on word embeddings
* training using the gate supervision thing

After it's working
* allow different batch_sizes for training and test
'''
