import sys

import yaml

from CryoEtchSimulator.utils.log import log_function_call  # noqa: E402
from CryoEtchSimulator.adsorption.simulator import AdsorptionSimulator
from CryoEtchSimulator.diffusion.simulator import DiffusionSimulator


@log_function_call
def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py input.yaml")
        sys.exit(1)
    path_inputs = sys.argv[1]
    with open(path_inputs, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.SafeLoader)

    adsRunner = AdsorptionSimulator(inputs)
    adsRunner.run()

    diffRunner = DiffusionSimulator(inputs, adsRunner.slab)
    diffRunner.run()


if __name__ == "__main__":
    main()
