import tensorflow as tf
from pathlib import Path

NUM_ITERATIONS = 5000
BATCH_SIZE = 100
TMP = 'tmp/model_0.2/'

i = 0
while Path(f'{TMP}{i}').exists():
    i += 1
model_dir = Path(f'{TMP}{i}')


def input_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        'data/train.csv', BATCH_SIZE, label_name='Survived'
    )
    return dataset


def validate_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        'data/train.csv', BATCH_SIZE,
        label_name='Survived',
        num_epochs=1
    )
    return dataset


tf.logging.set_verbosity(tf.logging.INFO)

feature_sex = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Sex',
    vocabulary_list=('male', 'female')
)

feature_pclass = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Pclass',
    vocabulary_list=(1, 2, 3)
)

feature_age = tf.feature_column.numeric_column(
    key='Age'
)

feature_embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Embarked',
    vocabulary_list=('C', 'Q', 'S')
)

feature_fare = tf.feature_column.numeric_column(
    key='Fare'
)

feature_parch = tf.feature_column.numeric_column(
    key='Parch'
)

feature_sib_sp = tf.feature_column.numeric_column(
    key='SibSp'
)

feature_columns = [
    tf.feature_column.indicator_column(feature_sex),
    tf.feature_column.indicator_column(feature_pclass),
    feature_age,
    tf.feature_column.indicator_column(feature_embarked),
    feature_parch,
    feature_sib_sp
]

config = tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=10)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[10, 10],
    feature_columns=feature_columns,
    n_classes=2,
    optimizer=tf.train.AdamOptimizer(0.0003),
    config=config
)

estimator.train(input_fn=input_fn, max_steps=NUM_ITERATIONS)
estimator.evaluate(input_fn=validate_fn, steps=1)
