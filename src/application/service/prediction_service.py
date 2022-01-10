from injector import Module, Binder, inject

from src.application.domain.service.dataset_service import DatasetService
from src.application.domain.service.training_service import TrainingService
from src.application.repository.config_repository import ConfigRepository
from src.application.repository.dataset_repository import DatasetRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl
from src.infrastructure.repository.dataset_repository_impl import DatasetRepositoryImpl


class PredictionService:
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

    def run(self, config_relative_path, input_filepath, weight_filepath, test):
        # 学習で使用した設定を読み込み
        train_config = self.config_repository.load(config_relative_path)

        # データセットをロード
        dataset = self.dataset_repository.load(input_filepath, train_config.delimiter, test)

        # 前処理
        label_encoder_config = self.config_repository.load_label_encoder(train_config.base_directory())
        input_data = self.dataset_service.preprocessing_for_prediction(dataset, train_config, label_encoder_config)

        # 予測
        outcome = self.training_service.predict(input_data, train_config, weight_filepath)

        # 結果を保存
        self.config_repository.save_outcome(train_config.base_directory(), dataset, outcome)


class PredictionServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(DatasetRepository, to=DatasetRepositoryImpl)
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)
