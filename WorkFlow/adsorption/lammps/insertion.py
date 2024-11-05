import subprocess
from pathlib import Path

import numpy as np

from utils.convert import pos2lmp
from utils.log import log_function_call  # noqa: E402
from adsorption.lammps.write_inputs import get_element_order  # noqa: E402
from adsorption.lammps.write_inputs import write_lmp_input_insertion  # noqa: E402


@log_function_call
def run(key, seed_max=1000000, **inputs):
    '''
    Repeat
        - Make directory
        - write lammps.in
        - write input.data
        - run lammps
    '''
    path_slab = inputs[key]["paths"]["slab"]
    path_mol = inputs[key]["paths"]["mol"]

    elem_order = get_element_order(path_mol, path_slab)

    n_insert = inputs[key]["mol_info"]["n_insert"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    path_lmp = inputs["relax"]["path_lmp_bin"]
    mol_name = inputs[key]["mol_info"]["name"]
    fix_height = inputs["constraint"]["fix_bottom_height"]
    run_short_MD = inputs[key]["mol_info"]["run_short_MD"]
    md_time = inputs[key]["mol_info"]["md_time"]
    md_temp = inputs[key]["mol_info"]["md_temp"]

    if key == "etchant":
        insert_global = True
    else:
        insert_global = False

    dst = inputs["dst"]
    path_dst = inputs[key]["paths"]["dst_1"]
    for i in range(n_repeat):
        dst = f'{path_dst}/{i}'
        p = Path(dst)
        if p.is_dir():
            continue
        p.mkdir(parents=True, exist_ok=True)
        if p.glob(f'lammps_{i}.out'):
            continue

        write_lmp_input_insertion(
            dst,
            path_mol,
            path_slab,
            mol_name,
            fix_height,
            n_insert,
            run_short_MD=run_short_MD,
            md_time=md_time,
            md_temp=md_temp,
            insert_global=insert_global,
        )

        dst_structure_input = f'{dst}/input.data'
        pos2lmp(path_slab, dst_structure_input, elem_order)

        seeds = np.random.randint(seed_max)
        cmd = f'{path_lmp} -in lammps.in -var SEEDS {seeds} > lammps_{i}.out'
        subprocess.run(cmd, cwd=p, shell=True)
