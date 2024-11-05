import os

from ase.io import read, write
from ase.constraints import FixAtoms

from utils.log import log_function_call  # noqa: E402


@log_function_call
def remove_etchant_molecule(key, idx_reconst_dict, **inputs):
    '''
    Remove etchant molecules and write new POSCAR to calculate
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    fix_height = inputs["constraint"]["fix_bottom_height"]
    path_dst = inputs[key]["paths"]["dst_3"]

    for i, idx_etchant in idx_reconst_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        poscar = read(src, index=-1)

        poscar_copy = poscar.copy()
        while poscar_copy:
            poscar_copy.pop()
        for idx, atom in enumerate(poscar):
            if idx in idx_etchant:
                continue
            poscar_copy.append(atom)

        poscar = poscar_copy
        c = FixAtoms([
            atom.index for atom in poscar if atom.position[2] <= fix_height
        ])
        poscar.set_constraint(c)
        dst = f'{path_dst}/{i}'
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)
        write(f'{dst}/POSCAR', poscar, format='vasp')
