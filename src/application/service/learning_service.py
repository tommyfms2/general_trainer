from injector import inject, Module, Binder

from src.application.domain.service.dataset_service import DatasetService
from src.application.domain.service.training_service import TrainingService
from src.application.repository.config_repository import ConfigRepository
from src.application.repository.dataset_repository import DatasetRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl
from src.infrastructure.repository.dataset_repository_impl import DatasetRepositoryImpl


class LearningService:
    @inject
    def __init__(self, dataset_repository: DatasetRepository,
                 config_repository: ConfigRepository,
                 dataset_service: DatasetService,
                 training_service: TrainingService
                 ):
        self.dataset_repository = dataset_repository
        self.config_repository = config_repository
        self.dataset_service = dataset_service
        self.training_service = training_service

    def run(self, config_relative_path, test):
        # 設定を読み込み
        train_config = self.config_repository.load(config_relative_path)

        # データセットをロード
        dataset = self.dataset_repository.load(train_config.input_filename, train_config.delimiter, test)

        # 前処理
        input_data, column_model_input_limits, label_encoder_config = \
            self.dataset_service.preprocessing(dataset, train_config)

        # 学習
        self.training_service.train(input_data, train_config, column_model_input_limits, label_encoder_config)


class LearningServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(DatasetRepository, to=DatasetRepositoryImpl)
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)
