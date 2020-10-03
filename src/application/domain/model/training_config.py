import pandas as pd

class TrainingConfig:

    @staticmethod
    def create_default_config(dataset: pd.DataFrame,
                              input_file_fullname: str,
                              output_dir_name: str,
                              delimiter: str,
                              batch_size: int,
                              dim_embed: int,
                              dim_hidden: int,
                              epochs: int,
                              target_column: str
                              ) -> dict:

        conf = {'input_file': input_file_fullname,
                'output_dir_name': output_dir_name,
                'delimiter': delimiter,
                'batch_size': batch_size,
                'dim_embed': dim_embed,
                'dim_hidden': dim_hidden,
                'epochs': epochs,
                'target_column': {'name': target_column, 'type': 'float'},
                'train_columns': []}

        headers = dataset.columns.values.tolist()

        if not target_column in headers:
            raise ValueError("There is not column name of ", target_column)

        for h in headers:
            if not h == target_column:
                conf_dict = {'name': h}
                if dataset[h].dtype == object:
                    if len(set(dataset[h].values)) < len(dataset[h].values) // 10:
                        conf_dict['encoder_type'] = 'label_encoder'
                    else:
                        conf_dict['encoder_type'] = 'text_encoder'
                        conf_dict['max_length'] = int(dataset[h].str.len().max())
                elif dataset[h].dtype == int or dataset[h].dtype == float:
                    column_min = float(dataset[h].min())
                    column_max = float(dataset[h].max())
                    if column_min < 0 or column_max >= 1000000:
                        conf_dict['encoder_type'] = 'text_encoder'
                        conf_dict['max_length'] = 10
                    elif dataset[h].dtype == int:
                        conf_dict['encoder_type'] = 'raw_encoder'
                        conf_dict['coeff'] = 1
                    else:
                        conf_dict['encoder_type'] = 'raw_encoder'
                        if column_max < 1000:
                            conf_dict['coeff'] = 100
                        else:
                            conf_dict['coeff'] = 1

                conf_dict['# appendix'] = \
                    'dtype: ' + str(dataset[h].dtype) + \
                    ', min: ' + str(dataset[h].dropna().min()) + \
                    ', max: ' + str(dataset[h].dropna().max()) + \
                    ', unique percent: ' + str(
                        100.0 * len(set(dataset[h].dropna().values)) / len(dataset[h].dropna().values))
                conf['train_columns'].append(conf_dict)

        return conf