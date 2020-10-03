from abc import ABCMeta, abstractmethod

import pandas as pd


class DatasetRepository(metaclass=ABCMeta):
    @abstractmethod
    def load_dataset(self, filename: str, delimiter: str) -> pd.DataFrame:
        raise NotImplementedError(self.__class__, " is not implemented.")
