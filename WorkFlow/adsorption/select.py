from pathlib import Path
import shutil

from utils.log import log_function_call  # noqa: E402


@log_function_call
def select_slab_with_minimal_energy(key, output, **inputs):
    energy_min = 0

    path_src = inputs[key]["paths"]["dst_2"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    max_step = inputs["relax"]["options"]["max_steps"]
    src = Path(path_src)
    for i in range(n_repeat):
        with open(src/f'{i}/thermo.dat', 'r') as f:
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
    p_dst = Path(path_dst)

    shutil.copy(src/f'{i_save}/POSCAR_relaxed', p_dst)
