import csv

import pandas as pd

from src.application.domain.model.value.model_config import ModelConfig
from src.application.repository.dataset_repository import DatasetRepository


class DatasetRepositoryImpl(DatasetRepository):

    def load_head(self, filename: str, sep: str) -> pd.DataFrame:
        print("### [TEST MODE] loading data...")
        rows = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader)
            i = 0
            for row in reader:
                i += 1
                if i > 200:
                    break
                rows.append(row)

        return pd.DataFrame(rows, columns=header)

    def load(self, filename: str, delimiter: str, head: bool) -> pd.DataFrame:
        delimiters = {'tab': '\t', 'comma': ','}
        if head:
            return self.load_head(filename, sep=delimiters[delimiter])

        print("### loading data...")
        return pd.read_csv(filename, sep=delimiters[delimiter], dtype='str', engine='python', error_bad_lines=False)
