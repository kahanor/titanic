# low-level-test-model: one layer, only uses 3 out of 10 features
import tensorflow as tf
import pandas as pd
import numpy as np


# used features: Pclass, Sex, Age,
# transform data into float32 values
def get_data_set(a_file):
    data_frame = pd.read_csv(a_file)
    data_set_shape = (data_frame.shape[0], 4)
    data_set = np.empty(data_set_shape)
    data_set[:, 0] = data_frame.loc[:, "Survived"]
    data_set[:, 1] = data_frame.loc[:, "Pclass"]
    data_set[:, 2] = data_frame.loc[:, "Age"]
    column_sex = np.empty(data_set_shape[0])
    data_frame_sex = data_frame.loc[:, "Sex"]
    for c in range(data_set_shape[0]-1):
        if data_frame_sex[c] == "male":
            column_sex[c] = 1
        else:
            column_sex[c] = 0

    data_set[:, 3] = column_sex
    return data_set


def get_next_batch(a_data_set):
    # TODO implement
    return 1


data_set = get_data_set("../data/train.csv")
print(data_set[0:10])
print(data_set.dtype)
print(data_set.shape)
exit(0)

# build model
X = tf.placeholder(tf.float32, shape=[None, 3, 1])
W = tf.Variable(tf.zeros([3, 1]))
b = tf.Variable(tf.zeros([1]))

Y = tf.nn.softmax(tf.matmul(X, W) + b)
Y_ = tf.placeholder(tf.float32, [None, 1])

# compute loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# reduce loss function
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# calc accuracy and loss function
is_correct = tf.equal(Y_, tf.round(Y))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# initialize session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# for loop
for i in range(1000):
    batch_X, batch_Y = (1, 2)  # get_next_batch(10)
    train_data = {X: batch_X, Y_: batch_Y}
    sess.run(train_step, feed_dict=train_data)
    if(i % 100 == 0):
        test_X, test_Y = (1, 2)  # get_test_data
        test_data = {X: test_X, Y_: test_Y}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print(f'{i} train accuracy: {a} cross_entropy: {c}')
    elif(i % 10 == 0):
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print(f'{i} test accuracy: {a} cross_entropy: {c}')
