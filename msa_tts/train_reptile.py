import argparse
import os
from .metatrainer import MetaTrainer
from .utils.generic import load_params
from .utils.grad_util import apply_grad, mix_grad


class Reptile(MetaTrainer):
    def __init__(self, **params):
        super(Reptile, self).__init__(**params)

    def run():
        pass


def main(args):
    params = load_params(os.path.join(args.params_path, "params.yml"))
    reptile = Reptile(**params)
    reptile.run()

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str)
    args = parser.parse_args()
    main(args)