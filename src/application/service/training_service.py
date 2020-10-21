from injector import inject, Module, Binder

from src.application.domain.model.cleaner import Cleaner
from src.application.domain.service.dataset_service import DatasetService
from src.application.repository.config_repository import ConfigRepository
from src.application.repository.dataset_repository import DatasetRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl
from src.infrastructure.repository.dataset_repository_impl import DatasetRepositoryImpl


class TrainingService:
    @inject
    def __init__(self, dataset_repository: DatasetRepository,
                 config_repository: ConfigRepository,
                 dataset_service: DatasetService):
        self.dataset_repository = dataset_repository
        self.config_repository = config_repository
        self.dataset_service = dataset_service

    def train(self, config_full_name, test):
        configs = self.config_repository.load(config_full_name)
        dataset = self.dataset_repository.load(configs['input_file'], configs['delimiter'], test)

        preprocessed_data, label_mapping = \
            self.dataset_service.label_by_config(dataset, configs['target_column']['name'], configs['train_columns'])







class TrainingServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(DatasetRepository, to=DatasetRepositoryImpl)
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)