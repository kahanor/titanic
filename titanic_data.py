import tensorflow as tf


class TitanicData(object):

    def __init__(self):
        pass

    def _encode_pclass(self, features, labels=None):
        features['Pclass'] = features['Pclass'] - 1
        if labels is None:
            return features
        return features, labels

    def load_train_data(
            self,
            train_file,
            batch_size,
            train_split
    ):
        dataset = tf.data.experimental.make_csv_dataset(
            train_file,
            batch_size=1,
            num_epochs=1,
            label_name='Survived',
        )

        dataset = dataset.map(self._encode_pclass)

        line_count = num_lines = sum(1 for line in open(train_file)) - 1
        train_lines = int(line_count * train_split)

        train_dataset = dataset.take(train_lines)
        valid_dataset = dataset.skip(train_lines)

        train_dataset = train_dataset.apply(
            tf.data.experimental.shuffle_and_repeat(10000)
        )
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        valid_dataset = valid_dataset.batch(batch_size)

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

    def train_input_fn(self):
        return self._train_dataset

    def valid_input_fn(self):
        return self._valid_dataset
