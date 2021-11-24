import numpy as np
import pandas as pd

from src.application.domain.model.entity.column import Column


class VolumeColumn(Column):

    def __init__(self, name: str, coeff: int):
        super().__init__(name)
        self.coeff = coeff

    def encode(self, datasets: pd.DataFrame, chara_label_dict=None) -> (dict, dict):
        tmp_data = datasets[self.name].fillna('0.0')
        tmp_data[tmp_data == ''] = '0.0'
        tmp_data[tmp_data == 'nan'] = '0.0'
        tmp_data = tmp_data.astype(float)
        max_volume = tmp_data.max() * float(self.coeff)
        datasets[self.name] = (tmp_data * float(self.coeff)).fillna(0).astype(int)
        minus_row_flag = datasets[self.name] < 0
        if True in minus_row_flag.tolist():
            print('AS_LABEL WARNING: There are some minus values that will be set 0')
            datasets[self.name][minus_row_flag] = 0
        limit_of_column = int(max_volume) + 1 if not np.isnan(max_volume) else 1
        encoded_method = {'max': limit_of_column}
        del tmp_data
        return encoded_method, {self.name: limit_of_column}

