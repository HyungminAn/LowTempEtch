from pathlib import Path
import shutil

from utils.log import log_function_call  # noqa: E402


@log_function_call
def select_slab_with_minimal_energy(key, output, **inputs):
    energy_min = 0

    path_src = inputs[key]["paths"]["dst_2"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    max_step = inputs["relax"]["max_steps"]
    for i in range(n_repeat):
        src = f'{path_src}/{i}'
        with open(f'{src}/thermo.dat', 'r') as f:
            last_line = f.readlines()[-1]

        step, energy, *_ = last_line.split()
        step = int(step)
        energy = float(energy)

        if step >= max_step:
            continue

        if energy <= energy_min:
            i_save = i
            energy_save = energy

    output["E_slab"] = energy_save

    path_dst = inputs[key]["paths"]["dst_3"]
    p_src = Path(f'{path_src}/{i_save}/POSCAR_relaxed')
    p_dst = Path(path_dst)

    shutil.copy(p_src, p_dst)
