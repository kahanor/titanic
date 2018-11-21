import tensorflow as tf

NUM_ITERATIONS = 15001
BATCH_SIZE = 100


def _encode_pclass(features, labels):
    features['Pclass'] = features['Pclass'] - 1
    return features, labels


def input_fn_train(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file, BATCH_SIZE, label_name='Survived',
        select_columns=['Sex', 'Pclass', 'Age', 'Survived']
    )
    dataset = dataset.map(_encode_pclass)
    return dataset


def input_fn_test(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file, BATCH_SIZE,
        select_columns=['Sex'], num_epochs=NUM_EPOCHS
    )
    return dataset


def make_hparam_str(learning_rate, hidden_units, dropout):
    return f'lr={learning_rate}, layers={hidden_units}, dropout={dropout}'


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

fare = tf.feature_column.numeric_column('Fare')

feature_columns = [sex, pclass, age, fare]


for learning_rate in [1E-3, 1E-4, 1E-5]:
    for hidden_units in [[], [5], [10], [5, 5], [10, 5]]:
        for dropout in [None, 0.2]:
            hparam_str = make_hparam_str(learning_rate, hidden_units, dropout)
            model_dir = f'/tmp/titanic/24/{hparam_str}'

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            estimator = tf.estimator.DNNClassifier(hidden_units=hidden_units,
                                                   feature_columns=[sex],
                                                   model_dir=model_dir,
                                                   optimizer=optimizer,
                                                   dropout=dropout)

            estimator.train(lambda: input_fn_train('train.csv'),
                            steps=NUM_ITERATIONS)
            accuracy = estimator.evaluate(lambda: input_fn_train('train.csv'))
