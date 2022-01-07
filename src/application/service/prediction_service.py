from injector import Module, Binder, inject

from src.application.repository.config_repository import ConfigRepository
from src.application.repository.dataset_repository import DatasetRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl
from src.infrastructure.repository.dataset_repository_impl import DatasetRepositoryImpl


class PredictionService:
    @inject
    def __init__(self, dataset_repository: DatasetRepository,
                 config_repository: ConfigRepository
                 ):
        self.dataset_repository = dataset_repository
        self.config_repository = config_repository

    def run(self, config_relative_path, test):
        # 設定を読み込み
        train_config = self.config_repository.load(config_relative_path)

        # データセットをロード
        dataset = self.dataset_repository.load(train_config.input_filename, train_config.delimiter, test)


class PredictionServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(DatasetRepository, to=DatasetRepositoryImpl)
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)
