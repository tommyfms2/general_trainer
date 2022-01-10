import os
import shutil
from collections import OrderedDict

import yaml
from numpy import ndarray
from pandas import DataFrame
from tensorflow import keras

from src.application.domain.model.entity.labelcolumn import LabelColumn
from src.application.domain.model.entity.target_column import TargetColumn
from src.application.domain.model.entity.textcolumn import TextColumn
from src.application.domain.model.entity.volumecolumn import VolumeColumn
from src.application.domain.model.value.model_config import ModelConfig
from src.application.domain.model.value.train_config import TrainConfig
from src.application.repository.config_repository import ConfigRepository


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


class ConfigRepositoryImpl(ConfigRepository):
    label_encoder_config_filename = 'label_encoder_config.yaml'
    model_config_filename = 'model.yaml'
    weight_filename = 'model_weights.hdf5'

    def dump(self, config_name: str, config: dict, force_rebuild: bool) -> None:
        if os.path.exists('./configs/' + config_name):
            if force_rebuild:
                shutil.rmtree('./configs/' + config_name)
                print('delete old directory')
            else:
                print('Abort: There is already the directory the name of which is the same as input.')
                return
        os.mkdir('./configs/' + config_name)
        print('made ', './configs/' + config_name)

        with open('./configs/' + config_name + "/conf.yml", "w") as f:
            yaml.dump(config, f, encoding='utf-8', allow_unicode=True, default_flow_style=False, sort_keys=False)

    def load(self, config_relative_path: str) -> TrainConfig:
        with open(config_relative_path, "r+") as f:
            configs = yaml.load(f)

        model_config = ModelConfig(
            configs['batch_size'],
            configs['dim_embed'],
            configs['dim_hidden'],
            configs['epochs']
        )

        train_column_configs = []
        for train_method in configs['train_columns']:
            if train_method['encoder_type'] == 'text_encoder':
                train_column_configs.append(TextColumn(train_method['name'], train_method['max_length']))
            elif train_method['encoder_type'] == 'label_encoder':
                train_column_configs.append(LabelColumn(train_method['name']))
            elif train_method['encoder_type'] == 'as_label':
                train_column_configs.append(VolumeColumn(train_method['name'], train_method['coeff']))

        # return model_config, Columns(column_list, TargetColumn(configs['target_column']['name']))
        return TrainConfig(model_config, config_relative_path, configs['input_file'], configs['output_dir_name'],
                           configs['delimiter'], TargetColumn(configs['target_column']['name']), train_column_configs)

    def reset_directory(self, base_directory: str, reset_directory_name) -> str:
        output_directory = base_directory + "/" + reset_directory_name

        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.mkdir(output_directory)

        return output_directory

    def reset_file(self, base_directory: str, reset_filename) -> str:
        output_filename = os.path.join(base_directory, reset_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        return output_filename

    def save_model(self, base_directory: str, model: keras.models.Model):
        open(os.path.join(base_directory, self.model_config_filename), 'w').write(model.to_yaml())

    def load_model(self, base_directory: str) -> keras.models.Model:
        with open(os.path.join(base_directory, self.model_config_filename), 'r') as f:
            yaml_string = f.read()

        return keras.models.model_from_yaml(yaml_string)

    def save_weight(self, base_directory: str, model: keras.models.Model):
        print('saving weights...')
        model.save_weights(os.path.join(base_directory, self.weight_filename))

    def load_weight(self, base_directory: str, filename: str, model: keras.models.Model) -> keras.models.Model:
        if filename == '':
            model.load_weights(os.path.join(base_directory, self.weight_filename))
        else:
            model.load_weights(filename)
        return model

    def copy(self, from_file: str, to_file) -> None:
        shutil.copy(from_file, to_file)

    def save_label_encoder(self, base_directory: str, content: dict):
        with open(base_directory + "/" + self.label_encoder_config_filename, "w", encoding='utf-8') as f:
            yaml.dump(content, f, encoding='utf-8', allow_unicode=True, default_flow_style=False,
                      sort_keys=False)

    def load_label_encoder(self, base_directory: str) -> dict:
        with open(base_directory + "/" + self.label_encoder_config_filename, "r+") as f:
            configs = yaml.load(f)

        return configs

    def save_outcome(self, base_directory: str, dataset: DataFrame, outcome: ndarray):
        dataset['outcome'] = outcome.tolist()
        dataset.to_csv(os.path.join(base_directory, 'outcome.csv'), index=False)
