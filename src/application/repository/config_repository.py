from abc import ABCMeta, abstractmethod


class ConfigRepository(metaclass=ABCMeta):
    @abstractmethod
    def dump(self, config_name: str, config: dict, force_rebuild: bool) -> None:
        raise NotImplementedError(self.__class__, " is not implemented.")

    def load(self, config_file_fullname: str) -> dict:
        raise NotImplementedError(self.__class__, " is not implemented.")
