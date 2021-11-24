from tensorflow import keras

from src.application.domain.model.value.model_config import ModelConfig


class KerasModelFactory:

    def __init__(self):
        None

    def build(self, model_config: ModelConfig, column_model_input_limits: list) -> keras.models.Model:
        dim = model_config.dim_embed
        hidden = model_config.dim_hidden

        flatten_layers = []
        inputs = []
        for i_n in column_model_input_limits:
            inputs_c = keras.layers.Input(shape=(1,))
            embed_c = keras.layers.Embedding(
                i_n,
                dim,
                input_length=1)(inputs_c)
            flatten_c = keras.layers.Flatten()(embed_c)

            inputs.append(inputs_c)
            flatten_layers.append(flatten_c)

        flatten = keras.layers.concatenate(flatten_layers)

        fc1 = keras.layers.Dense(hidden, activation='relu')(flatten)
        dp1 = keras.layers.Dropout(0.5)(fc1)

        outputs = keras.layers.Dense(1)(dp1)

        model = keras.models.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        model.summary()
        return model
