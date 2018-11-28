import pandas as pd
import tensorflow as tf


def train_split(
        train_in='data/train.csv',
        split=0.7,
        train_out='data/train_train.csv',
        valid_out='data/train_valid.csv'
):
    train_df = pd.read_csv(train_in)
    train_df = train_df.sample(frac=1)

    train_rows = int(len(train_df) * split)
    train = train_df.iloc[:train_rows]
    valid = train_df.iloc[train_rows:]

    train.to_csv(train_out)
    valid.to_csv(valid_out)


def write_predictions(
        prediction,
        in_file='data/test.csv',
        out_file='data/predict.csv'
):
    df_in = pd.read_csv(in_file)
    df_out = pd.DataFrame()
    df_out['PassengerId'] = df_in['PassengerId']
    survived = [pred['class_ids'][0] for pred in list(prediction)]
    df_out['Survived'] = pd.Series(survived)
    df_out.to_csv(out_file, index=False)
