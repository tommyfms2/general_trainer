from abc import ABCMeta, abstractmethod

import pandas as pd


class DatasetRepository(metaclass=ABCMeta):
    @abstractmethod
    def load(self, filename: str, delimiter: str, head: bool) -> pd.DataFrame:
        raise NotImplementedError(self.__class__, " is not implemented.")
