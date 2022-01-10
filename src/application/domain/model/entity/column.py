import pandas as pd


class Column:

    def __init__(self, name: str):
        self.name = name

    def encode(self, datasets: pd.DataFrame, chara_label_dict=None):
        None