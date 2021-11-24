import pandas as pd

from src.application.domain.model.entity.column import Column


class TextColumn(Column):

    def __init__(self, name: str, length: int):
        super().__init__(name)
        self.length = length

    def encode(self, datasets: pd.DataFrame, chara_label_dict=None) -> (dict, dict):
        limit_of_columns = {}
        for chara_idx in range(self.length):
            header = self.name + str(chara_idx)
            datasets[header] = \
                [chara_label_dict[x] for x in datasets[self.name].str[chara_idx].fillna('nan').values.tolist()]
            limit_of_columns[header] = len(chara_label_dict) + 1

        return {'mapping': "text_encoded"}, limit_of_columns

