from src.application.domain.model.encoder import Encoder


class Text(Encoder):

    def __init__(self, name: str, length: int):
        super().__init__(name)
        self.length = length

