from pathlib import Path
import os

import numpy as np
from ase.io import read
from ase.geometry import get_distances

from utils.log import log_function_call


def is_etchant_dissociated(
        path_rlx_trj, path_poscar_ref, bondLength_criteria=1.1):
    '''
    Check whether the inserted HF molecule has been dissoicated during relax

    TODO: generalization for molecules
    '''
    poscar_before_rlx = read(path_rlx_trj, index=0)
    poscar_ref = read(path_poscar_ref)
    idx_H, idx_F = find_HF_idx(poscar_before_rlx, poscar_ref)

    poscar_after_rlx = read(path_rlx_trj, index=-1)
    cell = poscar_after_rlx.get_cell()
    pos_after = poscar_after_rlx.get_positions()
    _, D_after = get_distances(pos_after, cell=cell, pbc=True)
    bondLength_after = D_after[idx_H, idx_F]

    if bondLength_after > bondLength_criteria:
        return True, (idx_H, idx_F)
    else:
        return False, (idx_H, idx_F)


def find_HF_idx(poscar, poscar_ref):
    '''
    Find the index of H atom in the inserted HF molecule
    by comparing the position of H atoms in *POSCAR* and *POSCAR_ref*
    '''

    pos = poscar.get_positions()
    cell = poscar.get_cell()
    _, D = get_distances(pos, p2=pos, cell=cell, pbc=True)

    pos_ref = poscar_ref.get_positions()
    _, D_ref = get_distances(pos_ref, p2=pos_ref, cell=cell, pbc=True)

    _, D_compare = get_distances(pos_ref, p2=pos, cell=cell, pbc=True)

    idx_H_all = np.array([atom.index for atom in poscar if atom.symbol == 'H'])
    dup_list = []

    for idx_H_ref in (atom.index for atom in poscar_ref if atom.symbol == 'H'):
        elem1 = find_2NN_symbol(idx_H_ref, poscar_ref, D_ref)

        for idx_H in idx_H_all[np.argsort(D_compare[idx_H_ref][idx_H_all])]:
            elem2 = find_2NN_symbol(idx_H, poscar, D)
            if elem1 == elem2:
                if idx_H in dup_list:
                    print('Already occupied!')
                    sys.exit(1)

                dup_list.append(idx_H)
                break

    idx_H_inserted = [i for i in idx_H_all if i not in dup_list][0]

    idx_F = np.array([atom.index for atom in poscar if atom.symbol == 'F'])
    idx_F_inserted = idx_F[np.argmin(D[idx_H_inserted][idx_F])]

    return idx_H_inserted, idx_F_inserted


def find_2NN_symbol(idx, poscar, D_len):
    '''
    For the given *poscar*, find the chemical symbol of the nearest neighbor
    of the atom with *idx*. Use *D_len* (distance matrix)
    '''
    idx_list = np.array([i for i in range(len(poscar)) if i != idx])
    idx_2NN = np.argmin(D_len[idx][idx_list])
    symbol = poscar[idx_list[idx_2NN]].symbol
    return symbol


@log_function_call
def check_etchant_dissociation(key, **inputs):
    '''
    classify results to physisorption and chemisorption,
    depending on whether the inserted etchant molecule has been dissociated.
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    path_slab = inputs[key]["paths"]["slab"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    format_rlx_trj = inputs["relax"]["path"]["traj"]
    idx_etchant_dict = {}

    for i in range(n_repeat):
        src = Path(f'{path_src}/{i}/{format_rlx_trj}')
        if not src.exists():
            continue
        cond, idx_etchant = is_etchant_dissociated(src, path_slab)
        if cond:
            continue
        idx_etchant_dict[i] = idx_etchant

    return idx_etchant_dict
