import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.application.domain.model.entity.column import Column


class LabelColumn(Column):

    def __init__(self, name: str):
        super().__init__(name)

    def encode(self, datasets: pd.DataFrame, chara_label_dict=None) -> (dict, dict):
        origin_values = datasets[self.name].fillna('null').values.tolist()
        datasets[self.name] = LabelEncoder().fit_transform(origin_values)
        num_of_labels = len(set(datasets[self.name]))
        column_encoder_mapping = {'mapping': dict(zip(
                ['nan'] + origin_values, [num_of_labels] + datasets[self.name].values.tolist()
            ))
        }
        del origin_values
        return column_encoder_mapping, {self.name: num_of_labels + 1}
