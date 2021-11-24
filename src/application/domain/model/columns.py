from src.application.domain.model.entity.target_column import TargetColumn


class Columns:

    def __init__(self, column_list: list, target_column: TargetColumn):
        self.column_list = column_list
        self.target_column = target_column
