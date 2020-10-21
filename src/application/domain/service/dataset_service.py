import pandas as pd


class DatasetService():

    def __init__(self):
        None

    def label_by_config(self, dataset: pd.DataFrame, target_col: str, train_methods: dict) -> (pd.DataFrame, dict):
        print("### preprocessing...")

        dataset = self.purge_rows_having_problems(dataset, target_col)



    def purge_rows_having_problems(self, dataset: pd.DataFrame, target_col: str) -> pd.DataFrame:
        dataset = dataset[~dataset[target_col].str.match('nan')]
        dataset = dataset[~(dataset[target_col] == '')]
        dataset = dataset[~dataset[target_col].isnull()]
        return dataset