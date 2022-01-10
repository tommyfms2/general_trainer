from typing import List

import pandas as pd
from tqdm import tqdm

from src.application.domain.model.entity.column import Column
from src.application.domain.model.entity.target_column import TargetColumn
from src.application.domain.model.entity.textcolumn import TextColumn
from src.application.domain.model.value.input_data import InputData
from src.application.domain.model.value.train_config import TrainConfig


class DatasetService:

    def __init__(self):
        None

    def preprocessing(self, dataset: pd.DataFrame, train_config: TrainConfig) -> (InputData, List[int], dict):

        dataset = self.purge_rows_having_problems(dataset, train_config.target_column,
                                                  train_config.train_column_configs)

        chara_label_dict = self.create_chara_labels(dataset, train_config.train_column_configs)

        label_encoder_config = {'text_encoded': chara_label_dict, 'encoder_dict': {}}
        column_limits = {}
        for train_column_config in tqdm(train_config.train_column_configs):
            # datasetを参照渡しで更新する
            encoded_method, limit_of_this_column = train_column_config.encode(dataset, chara_label_dict)
            label_encoder_config['encoder_dict'][train_column_config.name] = encoded_method
            column_limits.update(limit_of_this_column)

        label_encoder_config['column_model_input_limits'] = column_limits

        # 学習データの各カラムの最大入力値をもっておく
        column_model_input_limits = []
        columns_be_used = dataset.columns.tolist()
        for c_name in columns_be_used:
            if c_name in column_limits:
                column_model_input_limits.append(column_limits[c_name])

        return InputData(dataset), column_model_input_limits, label_encoder_config

    def preprocessing_for_prediction(self, dataset: pd.DataFrame, train_config: TrainConfig,
                                     label_encoder_config: dict) -> InputData:

        dataset = self.purge_rows_having_problems(dataset, train_config.target_column,
                                                  train_config.train_column_configs)

        for train_column_config in tqdm(train_config.train_column_configs):
            # datasetを参照渡しで更新する
            _, _ = train_column_config.encode(dataset, label_encoder_config['text_encoded'])
        return InputData(dataset)

    def purge_rows_having_problems(self, dataset: pd.DataFrame, target_column: TargetColumn,
                                   train_column_configs: List[Column]) -> pd.DataFrame:
        columns_to_use = [target_column.name]
        for column_config in train_column_configs:
            columns_to_use.append(column_config.name)

        # conf.ymlに無いcolumnを削除
        dataset_column_list = dataset.columns.tolist()
        for dataset_column in dataset_column_list:
            if dataset_column not in columns_to_use:
                del dataset[dataset_column]

        # 問題のあるデータを弾く
        # targetがNanのものを排除
        dataset = dataset[~dataset[target_column.name].str.match('nan')]
        dataset = dataset[~(dataset[target_column.name] == '')]
        dataset = dataset[~dataset[target_column.name].isnull()]
        return dataset

    def create_chara_labels(self, dataset: pd.DataFrame, train_column_configs: List[Column]) -> dict:
        charas = {}
        for train_column_config in train_column_configs:
            if isinstance(train_column_config, TextColumn):
                for c in range(train_column_config.length):
                    charas = set(
                        list(charas) + dataset[train_column_config.name].str[c].dropna(how='all').values.tolist()
                    )

        charas = set(list(charas) + ['nan'])
        charas_num_n = len(charas)
        return dict(zip(charas, range(charas_num_n)))
