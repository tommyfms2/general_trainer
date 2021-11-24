import argparse

from injector import Injector

from src.application.domain.service.training_service import TrainingServiceModule
from src.application.service.learning_service import LearningService, LearningServiceModule


def main(args):
    injector = Injector([LearningServiceModule(), TrainingServiceModule()])
    learning_service = injector.get(LearningService)

    learning_service.run(args.config, args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args)
