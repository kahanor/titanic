import pandas as pd
import tensorflow as tf
from os.path import splitext


def _append_to_file_name(file_name, append_str):
    file_name = splitex(file_name)
    return f'{file_name[0]}{append_str}{file_name[1]}'


def split_train_data(
        train_in='data/train.csv',
        split=0.7
):
    train_df = pd.read_csv(train_in)
    train_df = train_df.sample(frac=1)

    train_rows = int(len(train_df) * split)
    train = train_df.iloc[:train_rows]
    valid = train_df.iloc[train_rows:]

    train_out = _append_to_file_name(train_in, '_train')
    valid_out = _append_to_file_name(train_in, '_valid')
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
