import os
import shutil
from collections import OrderedDict

import yaml

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
