import argparse

from injector import Injector

from src.application.service.prediction_service import PredictionServiceModule, PredictionService


def main(args):
    injector = Injector([PredictionServiceModule()])
    prediction_service = injector.get(PredictionService)

    prediction_service.run(args.config, args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-w', '--weight', default='model_weights.hdf5')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args)
