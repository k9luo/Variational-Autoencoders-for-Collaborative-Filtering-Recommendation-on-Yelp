from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, load_yaml
from utils.modelnames import models

import argparse


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}
    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    hyper_parameter_tuning(R_train, R_valid, params, save_path=args.save_path, gpu_on=args.gpu)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--grid', dest='grid', default='config/vae.yml')
    parser.add_argument('--path', dest='path', default="data/yelp/")
    parser.add_argument('--save_path', dest='save_path', default="vae_tuning.csv")
    parser.add_argument('--train', dest='train', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid', default='Rvalid.npz')

    args = parser.parse_args()

    main(args)
