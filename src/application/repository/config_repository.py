from abc import ABCMeta, abstractmethod

from src.application.domain.model.value.train_config import TrainConfig


class ConfigRepository(metaclass=ABCMeta):
    @abstractmethod
    def dump(self, config_name: str, config: dict, force_rebuild: bool) -> None:
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def load(self, config_file_fullname: str) -> TrainConfig:
        raise NotImplementedError(self.__class__, " is not implemented.")
