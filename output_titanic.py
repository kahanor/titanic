import pandas as pd


def write_predictions(prediction, in_file='test.csv',
                      out_file='predict.csv'):
    df_in = pd.read_csv(in_file)
    df_out = pd.DataFrame()
    df_out['PassengerId'] = df_in['PassengerId']
    survived = [pred['class_ids'][0] for pred in list(prediction)]
    df_out['Survived'] = pd.Series(survived)
    df_out.to_csv(out_file, index=False)
