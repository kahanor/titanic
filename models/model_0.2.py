import tensorflow as tf
from pathlib import Path

NUM_EPOCHS = 1000
BATCH_SIZE = 100
PROJECT_ROOT = '/home/simon/Documents/Uni/Semester_03/KaggleLab/titanic'
TMP = f'{PROJECT_ROOT}/tmp/model_0.2/'

i = 0
while Path(f'{TMP}/{i}').exists():
    i += 1
model_dir = Path(f'{TMP}/{i}')


def input_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        f'{PROJECT_ROOT}/data/c_train.csv', BATCH_SIZE, label_name='Survived',
        select_columns=['Pclass', 'Sex', 'Survived',
                        'Age', 'Embarked', 'Parch', 'SibSp'],
        num_epochs=NUM_EPOCHS
    )
    return dataset


def validate_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        f'{PROJECT_ROOT}/data/c_validate.csv', BATCH_SIZE,
        label_name='Survived',
        select_columns=['Pclass', 'Sex', 'Survived',
                        'Age', 'Embarked', 'Parch', 'SibSp'],
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

estimator.train(input_fn=input_fn)
estimator.evaluate(input_fn=validate_fn, steps=1)
