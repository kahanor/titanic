import tensorflow as tf

NUM_EPOCHS = 5
BATCH_SIZE = 100


def nn_func(dataset):
    X = dataset['Pclass']
    Y = tf.layers.dense(X, tf.int32, activation='softmax')
    return Y


def _encode(features, labels):
    return features, labels


tf.enable_eager_execution()

dataset = tf.data.experimental.make_csv_dataset(
    'train.csv', BATCH_SIZE, label_name='Survived',
    select_columns=['Pclass', 'Sex', 'Age', 'Survived'], num_epochs=NUM_EPOCHS
)

for batch in dataset:
    print(batch)
