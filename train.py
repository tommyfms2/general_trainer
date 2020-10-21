import argparse

from injector import Injector

from src.application.service.training_service import TrainingServiceModule, TrainingService


def main(args):
    injector = Injector([TrainingServiceModule()])
    training_service = injector.get(TrainingService)

    training_service.train(args.config, args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args)

