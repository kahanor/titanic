import tensorflow as tf

NUM_EPOCHS = 5000
BATCH_SIZE = 100


def _encode_pclass(features, labels):
    features['Pclass'] = features['Pclass'] - 1
    return features, labels


def input_fn_train(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file, BATCH_SIZE, label_name='Survived',
        select_columns=['Sex', 'Pclass', 'Age', 'Survived'],
        num_epochs=NUM_EPOCHS
    )
    dataset = dataset.map(_encode_pclass)
    return dataset


def input_fn_test(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file, BATCH_SIZE,
        select_columns=['Sex'], num_epochs=NUM_EPOCHS
    )
    return dataset


def _encode(features, labels):
    return features, labels


# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


sex_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Sex',
    vocabulary_list=['male', 'female']
)
sex = tf.feature_column.indicator_column(sex_categorical)

pclass_categorical = tf.feature_column.categorical_column_with_identity(
    key='Pclass',
    num_buckets=3
)
pclass = tf.feature_column.indicator_column(pclass_categorical)

age = tf.feature_column.numeric_column('Age')

feature_columns = [sex, pclass, age]

estimator = tf.estimator.DNNClassifier(hidden_units=[10],
                                       feature_columns=[sex],
                                       model_dir='/tmp/titanic/16')

estimator.train(lambda: input_fn_train('train.csv'))
accuracy = estimator.evaluate(lambda: input_fn_train('train.csv'))
