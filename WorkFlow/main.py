from pathlib import Path
import sys
# To import other scripts
script_directory = Path(__file__).parent.absolute()
sys.path.append(str(script_directory))

import yaml

from utils.log import log_function_call  # noqa: E402
from adsorption.make_slab_additive import make_slab_with_additive  # noqa: E402
from adsorption.repeat_adsorption import repeat_adsorption  # noqa: E402
from adsorption.summarize import summarize_results  # noqa: E402


@log_function_call
def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py input.yaml")
        sys.exit(1)
    path_inputs = sys.argv[1]
    with open(path_inputs, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.SafeLoader)

    # Adsorption
    output = {}
    make_slab_with_additive(output, **inputs)
    repeat_adsorption(output, **inputs)
    summarize_results(output, **inputs)


if __name__ == "__main__":
    main()
