from abc import ABCMeta, abstractmethod

from numpy import ndarray
from pandas import DataFrame
from tensorflow import keras

from src.application.domain.model.value.train_config import TrainConfig


class ConfigRepository(metaclass=ABCMeta):
    @abstractmethod
    def dump(self, config_name: str, config: dict, force_rebuild: bool) -> None:
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def load(self, config_file_fullname: str) -> TrainConfig:
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def reset_directory(self, base_directory: str, reset_directory_name) -> str:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def reset_file(self, base_directory: str, reset_filename) -> str:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def save_model(self, base_directory: str, model: keras.models.Model):
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def load_model(self, base_directory: str) -> keras.models.Model:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def save_weight(self, base_directory: str, model: keras.models.Model):
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def load_weight(self, base_directory: str, filename: str, model: keras.models.Model) -> keras.models.Model:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def copy(self, from_file: str, to_file) -> None:
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def save_label_encoder(self, base_directory: str, content: dict):
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def load_label_encoder(self, base_directory: str) -> dict:
        raise NotImplementedError(self.__class__, " is not implemented.")

    @abstractmethod
    def save_outcome(self, base_directory: str, dataset: DataFrame, outcome: ndarray):
        raise NotImplementedError(self.__class__, " is not implemented.")
