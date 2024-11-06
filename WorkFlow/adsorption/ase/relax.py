import os
from pathlib import Path
import shutil

from ase.io import write

from utils.convert import lmp2pos  # noqa: E402
from utils.log import log_function_call  # noqa: E402
from adsorption.perturb import set_perturbation  # noqa: E402

from relax.relax_sevenn_d3 import relax  # noqa: E402


@log_function_call
def run(key, **inputs):
    '''
    Repeat
        - Make directory
        - copy input.yaml
        - run ASE_relax
    '''
    path_slab = inputs[key]["paths"]["slab"]

    n_repeat = inputs[key]["mol_info"]["n_repeat"]

    dst = inputs["dst"]
    path_src = inputs[key]["paths"]["dst_1"]
    path_dst = inputs[key]["paths"]["dst_2"]
    fix_bottom_height = inputs["constraint"]["fix_bottom_height"]

    perturb_flag = inputs[key]["perturb"]["flag"]

    for i in range(n_repeat):
        dst = f'{path_dst}/{i}'
        dst_poscar = f'{dst}/POSCAR'

        p = Path(dst)
        poscar_relaxed = p / inputs["relax"]["path"]["output"]
        if poscar_relaxed.exists():
            continue
        p.mkdir(parents=True, exist_ok=True)

        src = f'{path_src}/{i}'
        src_lmp_dat = f'{src}/FINAL.coo'
        lmp2pos(
            src_lmp_dat,
            dst_poscar,
            fix_bottom_height=fix_bottom_height,
        )

        is_relax_success = relax(dst_poscar, dst, **inputs)

    if not perturb_flag:
        return

    for i in range(n_repeat):
        dst = f'{path_dst}/{i}'
        dst_poscar = f'{dst}/POSCAR'
        rlx_traj = f'{dst}/' + inputs["path"]["traj"]
        scale = inputs[key]["perturb"]["scale"]
        cutoff = inputs[key]["perturb"]["cutoff"]
        poscar_perturb = set_perturbation(
            rlx_traj,
            path_slab,
            scale,
            cutoff,
            fix_bottom_height,
        )
        os.makedirs(f'{dst}/initial', exist_ok=True)
        for file in os.listdir(dst):
            shutil.move(f'{dst}/{file}', f'{dst}/initial')
        write(dst_poscar, poscar_perturb, format='vasp')
        is_relax_success = relax(dst_poscar, dst, **inputs)
