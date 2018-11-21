import tensorflow as tf
import pandas as pd

NUM_ITERATIONS = 10001
BATCH_SIZE = 100


def _encode_pclass(features, labels=None):
    features['Pclass'] = features['Pclass'] - 1
    if labels is None:
        return features
    return features, labels


def input_fn_train(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file,
        BATCH_SIZE,
        label_name='Survived',
        select_columns=['Sex', 'Pclass', 'Age',
                        'Fare', 'SibSp', 'Parch', 'Survived']
    )
    dataset = dataset.map(_encode_pclass)
    return dataset


def input_fn_eval(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file,
        BATCH_SIZE,
        label_name='Survived',
        select_columns=['Sex', 'Pclass', 'Age',
                        'Fare', 'SibSp', 'Parch', 'Survived'],
        num_epochs=1
    )
    dataset = dataset.map(_encode_pclass)
    return dataset


def input_fn_test(csv_file):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file,
        BATCH_SIZE,
        select_columns=['Sex', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch'],
        num_epochs=1,
        shuffle=False
    )
    dataset = dataset.map(_encode_pclass)
    return dataset


def make_hparam_str(learning_rate, hidden_units, dropout):
    return f'lr={learning_rate}, layers={hidden_units}, dropout={dropout}'


def write_predictions(prediction, in_file='test.csv', out_file='predict.csv'):
    df_in = pd.read_csv(in_file)
    df_out = pd.DataFrame()
    df_out['PassengerId'] = df_in['PassengerId']
    survived = [pred['class_ids'][0] for pred in list(prediction)]
    df_out['Survived'] = pd.Series(survived)
    df_out.to_csv(out_file, index=False)


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

sib_sb = tf.feature_column.numeric_column('SibSp')

par_ch = tf.feature_column.numeric_column('Parch')

feature_columns = [sex, pclass, age, fare, sib_sb, par_ch]


for learning_rate in [1E-3, 1E-4, 1E-5]:
    for hidden_units in [[], [5], [10], [5, 5], [10, 5]]:
        dropout = None

        hparam_str = make_hparam_str(learning_rate, hidden_units, dropout)
        model_dir = f'/tmp/titanic/02/{hparam_str}'
        config = tf.estimator.RunConfig(model_dir=model_dir,
                                        save_summary_steps=100)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        estimator = tf.estimator.DNNClassifier(
            hidden_units=hidden_units,
            feature_columns=feature_columns,
            model_dir=model_dir,
            optimizer=optimizer,
            dropout=dropout
        )

        estimator.train(lambda: input_fn_train('train.csv'),
                        steps=NUM_ITERATIONS)
        accuracy = estimator.evaluate(lambda: input_fn_eval('train.csv'))
        prediction = estimator.predict(lambda: input_fn_test('test.csv'))

        out_file = f'predict_{hparam_str}.csv'
        write_predictions(prediction, out_file=out_file)
