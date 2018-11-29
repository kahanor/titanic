import tensorflow as tf
from pathlib import Path
from output_titanic import write_predictions

NUM_ITERATIONS = 5000
BATCH_SIZE = 100
TMP = 'tmp/model_0.3/'


def increment_folder_number():
    i = 0
    while Path(f'{TMP}{i}').exists():
        i += 1
    model_dir = Path(f'{TMP}{i}')


def build_model_dir(hidden_units, learn_rate, features):
    result_string = "hu"
    for entry in hidden_units:
        result_string += f'-{entry}'

    result_string += "_lr"
    result_string += f'-{str(learn_rate)[2:]}'

    result_string += "_f"
    for entry in features:
        result_string += f'-{entry}'

    return result_string


def input_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        'data/c_train.csv',
        BATCH_SIZE,
        label_name='Survived',
        shuffle=True
    )
    return dataset


def validate_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        'data/c_validate.csv', BATCH_SIZE,
        label_name='Survived',
        num_epochs=1
    )
    return dataset


def predict_fn():
    dataset = tf.data.experimental.make_csv_dataset(
        'data/test.csv', BATCH_SIZE,
        num_epochs=1
    )
    return dataset


def build_feature_columns(config=['Sex', 'Pclass']):
    feature_columns = []

    if 'Sex' in config:
        feature_sex = tf.feature_column.categorical_column_with_vocabulary_list(
            key='Sex',
            vocabulary_list=('male', 'female')
        )
        feature_columns.append(tf.feature_column.indicator_column(feature_sex))

    if 'Pclass' in config:
        feature_pclass = tf.feature_column.categorical_column_with_vocabulary_list(
            key='Pclass',
            vocabulary_list=(1, 2, 3)
        )
        feature_columns.append(tf.feature_column.indicator_column(feature_pclass))

    if 'Age' in config:
        feature_age = tf.feature_column.numeric_column(
            key='Age'
        )
        feature_columns.append(feature_age)

    if 'Embarked' in config:
        feature_embarked = tf.feature_column.categorical_column_with_vocabulary_list(
            key='Embarked',
            vocabulary_list=('C', 'Q', 'S')
        )
        feature_columns.append(tf.feature_column.indicator_column(feature_embarked))

    if 'Fare' in config:
        feature_fare = tf.feature_column.numeric_column(
            key='Fare'
        )
        feature_columns.append(feature_fare)

    if 'Parch' in config:    
        feature_parch = tf.feature_column.numeric_column(
            key='Parch'
        )
        feature_columns.append(feature_parch)

    if 'SibSp' in config:
        feature_sib_sp = tf.feature_column.numeric_column(
            key='SibSp'
        )
        feature_columns.append(feature_sib_sp)

    return feature_columns


def run(warm_start=False,
        eval=False,
        pred=False,
        hidden_units=[5],
        learn_rate=0.0003,
        features=['Sex', 'Pclass']
        ):

    feature_columns = build_feature_columns(features)

    model_dir = f'{TMP}/{build_model_dir(hidden_units, learn_rate, features)}'

    warm_start_from = None
    if warm_start is True:
        warm_start_from = f'{model_dir}/checkpoint'

    config = tf.estimator.RunConfig(model_dir=model_dir, save_summary_steps=10)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=feature_columns,
        n_classes=2,
        optimizer=tf.train.AdamOptimizer(learn_rate),
        config=config,
        warm_start_from=warm_start_from
    )

    if warm_start_from is None:
        estimator.train(input_fn=input_fn, max_steps=NUM_ITERATIONS)

    if eval:
        estimator.evaluate(input_fn=validate_fn, steps=1)

    if pred:
        pred = estimator.predict(input_fn=predict_fn)
        write_predictions(
            prediction=pred,
            in_file='data/test.csv',
            out_file=f'{model_dir}/predict.csv'
            )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    hidden_units_grid = [
        [2],
        [5],
        [10],
        [2, 2],
        [5, 5],
        [10, 10]
        ]

    learn_rate_grid = [0.0003, 0.001]

    feature_set_grid = [
        ['Sex', 'Pclass'],
        ['Sex', 'Pclass', 'Age']
        ]

    for hidden_units in hidden_units_grid:
        for learn_rate in learn_rate_grid:
            for feature_set in feature_set_grid:
                run(warm_start=False,
                    eval=True,
                    pred=True,
                    hidden_units=hidden_units,
                    learn_rate=learn_rate,
                    features=feature_set
                    )
