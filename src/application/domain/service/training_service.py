from typing import List

from injector import Module, Binder, inject
from tensorflow import keras

from src.application.domain.model.value.input_data import InputData
from src.application.domain.model.value.train_config import TrainConfig
from src.application.domain.service.keras_model_factory import KerasModelFactory
from src.application.repository.output_reository import OutputRepository
from src.infrastructure.repository.output_repository_impl import OutputRepositoryImpl


class TrainingService:
    @inject
    def __init__(self, keras_model_factory: KerasModelFactory,
                 output_repository: OutputRepository
                 ):
        self.keras_model_factory = keras_model_factory
        self.output_repository = output_repository

    def run(self, input_data: InputData, train_config: TrainConfig, column_model_input_limits: List[int]):
        model = self.keras_model_factory.build(train_config.model_config, column_model_input_limits)

        (train_x, train_y, test_x, test_y, val_x, val_y) = input_data.create_x_t(train_config.target_column)

        # repositoryでoutput_dディレクトリを作成してパスをもらう
        output_directory = self.output_repository.reset_directory(train_config.base_directory(),
                                                                  train_config.output_dir_name)
        # repositoryでsnapshotsをからにしてパスをもらう
        snapshots_directory = self.output_repository.reset_directory(output_directory, 'snapshots')

        # private methodでcbを返却する
        cb = self.make_callbacks(snapshots_directory)

        # モデル形式を保存する
        self.output_repository.save_string(output_directory, 'model.yaml', model.to_yaml())

        # 学習する
        model.fit(
            train_x,
            train_y,
            batch_size=train_config.model_config.batch_size,
            epochs=train_config.model_config.epochs,
            verbose=1,
            shuffle=True,
            validation_data=[test_x, test_y],
            callbacks=[cb]
        )

        # 結果を保存する
        self.output_repository.save_weight(output_directory, 'model_weights.hdf5', model)

    def make_callbacks(self, snapshot_directory: str):
        model_f = self.output_repository.reset_file(snapshot_directory,
                                                    'ss_epoch{epoch:02d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5')
        return keras.callbacks.ModelCheckpoint(filepath=model_f, monitor='val_loss', verbose=1, save_best_only=True,
                                               mode='min', period=2)


class TrainingServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(OutputRepository, to=OutputRepositoryImpl)
