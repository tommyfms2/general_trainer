from typing import List

from src.application.domain.model.entity.column import Column
from src.application.domain.model.entity.target_column import TargetColumn
from src.application.domain.model.value.model_config import ModelConfig


class TrainConfig:

    def __init__(self, model_config: ModelConfig, config_relative_path: str, input_filename: str, output_dir_name: str,
                 delimiter: str, target_column: TargetColumn, train_column_configs: List[Column]):
        self.model_config = model_config
        self.config_relative_path = config_relative_path
        self.input_filename = input_filename
        self.output_dir_name = output_dir_name
        self.delimiter = delimiter
        self.target_column = target_column
        self.train_column_configs = train_column_configs

    def only_train_column_limits(self, column_limits: dict) -> list:
        column_model_input_limits = []
        for column, limits in column_limits.items():
            if column != self.target_column:
                column_model_input_limits.append(limits)
        return column_model_input_limits

    def base_directory(self) -> str:
        return "/".join(self.config_relative_path.split("/")[:-1])
