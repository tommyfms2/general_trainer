import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.application.domain.model.entity.target_column import TargetColumn


class InputData:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_x_t(self, target_column: TargetColumn):
        threshold = self.data.shape[0] // 10 * 9
        columns_to_use = self.data.columns.tolist()
        columns_to_use.remove(target_column.name)

        X = self.data[:threshold][columns_to_use].astype(int).values
        X_v = self.data[threshold:][columns_to_use].astype(int).values
        y = self.data[:threshold][target_column.name].astype(float).values
        y_v = self.data[threshold][target_column.name].astype(float).values

        seed = 1
        np.random.seed(seed)
        skf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break

        X_train = [X_train[:, i] for i in range(X.shape[1])]
        X_test = [X_test[:, i] for i in range(X.shape[1])]
        X_val = [X_v[:, i] for i in range(X_v.shape[1])]

        return (X_train, y_train, X_test, y_test, X_val, y_v)



