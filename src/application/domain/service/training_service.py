from typing import List

import numpy as np
from injector import Module, Binder, inject
from numpy import ndarray
from tensorflow import keras

from src.application.domain.model.value.input_data import InputData
from src.application.domain.model.value.train_config import TrainConfig
from src.application.domain.service.keras_model_factory import KerasModelFactory
from src.application.repository.config_repository import ConfigRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl


class TrainingService:
    @inject
    def __init__(self, keras_model_factory: KerasModelFactory,
                 config_repository: ConfigRepository
                 ):
        self.keras_model_factory = keras_model_factory
        self.config_repository = config_repository

    def train(self, input_data: InputData, train_config: TrainConfig, column_model_input_limits: List[int],
              label_encoder_config: dict):
        model = self.keras_model_factory.build(train_config.model_config, column_model_input_limits)

        (train_x, train_y, test_x, test_y, val_x, val_y) = input_data.create_x_t(train_config.target_column)

        # repositoryでoutput_dディレクトリを作成してパスをもらう
        output_directory = self.config_repository.reset_directory(train_config.base_directory(),
                                                                  train_config.output_dir_name)
        # repositoryでsnapshotsをからにしてパスをもらう
        snapshots_directory = self.config_repository.reset_directory(output_directory, 'snapshots')

        # private methodでcbを返却する
        cb = self.make_callbacks(snapshots_directory)

        # モデル形式を保存する
        self.config_repository.save_model(output_directory, model)

        # 学習する
        model.fit(
            train_x,
            train_y,
            batch_size=train_config.model_config.batch_size,
            epochs=train_config.model_config.epochs,
            verbose=1,
            shuffle=True,
            validation_data=(test_x, test_y),
            callbacks=[cb]
        )

        # 結果を保存する
        self.config_repository.save_weight(output_directory, model)

        # エンコード設定を保存する
        self.config_repository.save_label_encoder(output_directory, label_encoder_config)

        # 学習設定ファイルをコピーする
        self.config_repository.copy(train_config.config_relative_path, output_directory + "/conf.yaml")

    def make_callbacks(self, snapshot_directory: str):
        model_f = self.config_repository.reset_file(snapshot_directory,
                                                    'ss_epoch{epoch:02d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5')
        return keras.callbacks.ModelCheckpoint(filepath=model_f, monitor='val_loss', verbose=1, save_best_only=True,
                                               mode='min', period=2)

    def predict(self, input_data: InputData, train_config: TrainConfig, weight_filepath: str) -> ndarray:
        model = self.config_repository.load_model(train_config.base_directory())
        model = self.config_repository.load_weight(train_config.base_directory(), weight_filepath, model)

        pred_x = input_data.create_x_pred(train_config.target_column)

        return np.array(model.predict(pred_x, batch_size=1024 * 8)).flatten()


class TrainingServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)
