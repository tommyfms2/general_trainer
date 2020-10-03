import argparse

from injector import Injector

from src.application.service.config_builder_service import ConfigBuilderService, ConfigBuilderServiceModule


def main(args):
    injector = Injector([ConfigBuilderServiceModule()])
    config_builder_service = injector.get(ConfigBuilderService)

    config_builder_service.build(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', required=True)
    parser.add_argument('-i', '--input_file_fullname', required=True)
    parser.add_argument('-t', '--target_column', required=True)
    parser.add_argument('-o', '--output_dir_name', default='outputs')
    parser.add_argument('-d', '--delimiter', default='tab')
    parser.add_argument('-b', '--batch_size', default=8192, type=int)
    parser.add_argument('-de', '--dim_embed', default=64, type=int)
    parser.add_argument('-dh', '--dim_hidden', default=128, type=int)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('--force_rebuild', action='store_true')
    args = parser.parse_args()

    main(args)
