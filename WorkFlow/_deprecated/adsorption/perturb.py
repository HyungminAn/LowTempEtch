import sys

import numpy as np
from ase.io import read
from ase.geometry import get_distances
from ase.constraints import FixAtoms


def find_2NN_symbol(idx, poscar, D_len):
    '''
    For the given *poscar*, find the chemical symbol of the nearest neighbor
    of the atom with *idx*. Use *D_len* (distance matrix)
    '''
    idx_list = np.array([i for i in range(len(poscar)) if i != idx])
    idx_2NN = np.argmin(D_len[idx][idx_list])
    symbol = poscar[idx_list[idx_2NN]].symbol
    return symbol


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


def set_perturbation(
        path_rlx_trj, path_poscar_slab, scale, cutoff, fix_bottom_height):
    '''
    Set the perturbation for the inserted HF molecule
    '''
    idx_H, idx_F = find_HF_idx(
            read(path_rlx_trj, index=0),
            read(path_poscar_slab),
            )

    poscar = read(path_rlx_trj, index=-1)
    pos = poscar.get_positions()
    cell = poscar.get_cell()
    _, D = get_distances(pos, p2=pos, cell=cell, pbc=True)

    idx_to_perturb = [idx for idx, d in enumerate(D[idx_H]) if d < cutoff]
    for idx in idx_to_perturb:
        if idx == idx_H or idx == idx_F:
            pos[idx] -= scale
        else:
            pos[idx] += scale * np.random.rand(3)

    poscar.set_positions(pos)

    c = FixAtoms(indices=[
        atom.index for atom in poscar
        if atom.position[2] < fix_bottom_height]
        )
    poscar.set_constraint(c)

    return poscar
