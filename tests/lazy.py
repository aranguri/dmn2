import tensorflow as tf

X1 = tf.Variable(1.)
X2 = tf.Variable(1.)

cond_value = tf.Variable(False)

loss = X1
optimizer = tf.train.AdamOptimizer(learning_rate=1)

def result_1():
    minimize = optimizer.minimize(loss)
    # assign_1 = tf.assign(X1, 2.)
    with tf.control_dependencies([minimize]):
        return tf.identity(X1)

def result_2():
    assign_2 = tf.assign(X2, 2.)
    with tf.control_dependencies([assign_2]):
        return tf.identity(X2)

cond_result = tf.cond(cond_value, result_1, result_2)

with tf.Session() as sesh:
    sesh.run(tf.initialize_all_variables())
    sesh.run(cond_result)
    print(sesh.run(X1))

'''
import tensorflow as tf



def true_fn():
    with tf.control_dependencies([minimize]):
        return tf.identity(loss)

def false_fn():
    return tf.identity(loss)

loss = tf.cond(tf.constant(1) > tf.constant(0), true_fn, false_fn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        loss_ = sess.run([loss])
        print(loss_)
'''
