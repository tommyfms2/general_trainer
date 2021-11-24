import os
import shutil
from collections import OrderedDict

import yaml

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

        # column_list = []
        # for column_raw in configs['train_columns']:
        #     encoder_type = column_raw['encoder_type']
        #     if encoder_type == 'text_encoder':
        #         column_list.append(TextColumn(column_raw['name'], column_raw['max_length']))
        #     elif encoder_type == 'label_encoder':
        #         column_list.append(LabelColumn(column_raw['name']))
        #     elif encoder_type == 'as_labels':
        #         column_list.append(VolumeColumn(column_raw['name'], column_raw['coeff']))

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
