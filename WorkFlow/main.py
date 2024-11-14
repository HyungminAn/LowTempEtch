from pathlib import Path
import sys
# To import other scripts
script_directory = Path(__file__).parent.absolute()
sys.path.append(str(script_directory))

import yaml

from utils.log import log_function_call  # noqa: E402
from adsorption.simulator import AdsorptionSimulator
from diffusion.simulator import DiffusionSimulator


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
