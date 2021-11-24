from tensorflow import keras

from abc import ABCMeta, abstractmethod


class OutputRepository(metaclass=ABCMeta):
    @abstractmethod
    def reset_directory(self, base_directory: str, reset_directory_name) -> str:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def reset_file(self, base_directory: str, reset_filename) -> str:
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def save_string(self, base_directory: str, filename: str, content: str):
        raise NotImplementedError(self.__class__, "is not implemented.")

    @abstractmethod
    def save_weight(self, base_directory: str, filename: str, model: keras.models.Model):
        raise NotImplementedError(self.__class__, "is not implemented.")
