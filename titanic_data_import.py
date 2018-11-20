import tensorflow as tf

NUM_EPOCHS = 5

dataset = tf.data.TextLineDataset(['train.csv'])

dataset = dataset.shuffle(100000)
dataset = dataset.repeat(NUM_EPOCHS)
