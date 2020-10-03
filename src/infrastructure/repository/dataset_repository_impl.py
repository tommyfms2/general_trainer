import pandas as pd

from src.application.repository.dataset_repository import DatasetRepository


class DatasetRepositoryImpl(DatasetRepository):
    def load_dataset(self, filename: str, delimiter: str) -> pd.DataFrame:
        delimiters = {'tab': '\t', 'comma': ','}
        return pd.read_csv(filename, sep=delimiters[delimiter], dtype='str', engine='python', error_bad_lines=False)
