import tensorflow as tf
import pandas as pd
import titanic_data as t_data

NUM_ITERATIONS = 10001


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

sib_sb = tf.feature_column.numeric_column('SibSp')

par_ch = tf.feature_column.numeric_column('Parch')

feature_columns = [sex, pclass, age, fare, sib_sb, par_ch]

train_csv, valid_csv = t_data.split_train_data()


for learning_rate in [1E-3, 1E-4, 1E-5]:
    for hidden_units in [[], [5], [10], [5, 5], [10, 5]]:
        dropout = None

        hparam_str = make_hparam_str(learning_rate, hidden_units, dropout)
        model_dir = f'/tmp/titanic/01/{hparam_str}'
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
        estimator.train(lambda: t_data.input_fn_train(train_csv),
                        steps=NUM_ITERATIONS)
        accuracy = estimator.evaluate(
            lambda: t_data.input_fn_eval(valid_csv))
        prediction = estimator.predict(
            lambda: t_data.input_fn_test('data/test.csv'))

        out_file = f'predict_{hparam_str}.csv'
        t_data.write_predictions(prediction, out_file=out_file)
