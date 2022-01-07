import os
import shutil

import yaml
from tensorflow import keras

from src.application.repository.output_reository import OutputRepository


class OutputRepositoryImpl(OutputRepository):

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

    def save_string(self, base_directory: str, filename: str, content: str):
        open(os.path.join(base_directory, filename), 'w').write(content)

    def save_weight(self, base_directory: str, filename: str, model: keras.models.Model):
        print('saving weights...')
        model.save_weights(os.path.join(base_directory, filename))

    def copy(self, from_file: str, to_file) -> None:
        shutil.copy(from_file, to_file)

    def save_yaml(self, base_directory: str, filename: str, content: dict):
        with open(base_directory + "/" + filename, "w", encoding='utf-8') as f:
            yaml.dump(content, f, encoding='utf-8', allow_unicode=True, default_flow_style=False,
                      sort_keys=False)
