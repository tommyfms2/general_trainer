from injector import inject, Module, Binder

from src.application.service.training_config_factory import TrainingConfigFactory
from src.application.repository.config_repository import ConfigRepository
from src.application.repository.dataset_repository import DatasetRepository
from src.infrastructure.repository.config_repository_impl import ConfigRepositoryImpl
from src.infrastructure.repository.dataset_repository_impl import DatasetRepositoryImpl


class ConfigBuilderService:
    @inject
    def __init__(self, dataset_repository: DatasetRepository, config_repository: ConfigRepository):
        self.dataset_repository = dataset_repository
        self.config_repository = config_repository

    def build(self, args):
        dataset = self.dataset_repository.load(args.input_file_fullname, args.delimiter, False)

        training_config = TrainingConfigFactory.create_default_config(
            dataset=dataset,
            input_file_fullname=args.input_file_fullname,
            output_dir_name=args.output_dir_name,
            delimiter=args.delimiter,
            batch_size=args.batch_size,
            dim_embed=args.dim_embed,
            dim_hidden=args.dim_hidden,
            epochs=args.epochs, target_column=args.target_column
        )

        self.config_repository.dump(args.config_name, training_config, args.force_rebuild)


class ConfigBuilderServiceModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(DatasetRepository, to=DatasetRepositoryImpl)
        binder.bind(ConfigRepository, to=ConfigRepositoryImpl)
