import tensorflow as tf
# from preprocess import BabiTask
from model import DMNCell
import numpy as np

batch_size = 32
embeddings_size = 50 #word embeddings dimensionality
eos_vector = np.random.randn(embeddings_size,) #Todo: later on get it from babi

dmn_cell = DMNCell(eos_vector, input_h_size=32, question_h_size=32, episode_h_size=32, similarity_layer_size=32, num_passes=3)
# babi_task = BabiTask(task_num=1, batch_size)

input = tf.placeholder(tf.float32, shape=(batch_size, None, embeddings_size))
question = tf.placeholder(tf.float32, shape=(batch_size, None, embeddings_size))
answer = tf.placeholder(tf.float32, shape=(batch_size, embeddings_size))

output = dmn_cell.run(input, question)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    input_ = np.random.randn(batch_size, 20, embeddings_size) #NOTe: EOS isn't going to work
    question_ = np.random.randn(batch_size, 4, embeddings_size)
    answer_ = np.random.randn(batch_size, embeddings_size)
    # input_, question_, answer_, label = babi_task.next_batch()
    output_ = sess.run([output], feed_dict={input: input_, question: question_, answer: answer_})
    print(output_)
