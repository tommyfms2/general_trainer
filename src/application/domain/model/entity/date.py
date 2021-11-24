from src.application.domain.model.entity.column import Column


class Date(Column):

    def __init__(self, name: str):
        super().__init__(name)
