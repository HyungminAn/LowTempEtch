import numpy as np

from ase.io import read, write
from ase.geometry import get_distances


def remove_slab_F(poscar, D_len):
    idx_F = [atom.index for atom in poscar if atom.symbol == 'F']
    idx_Si = [atom.index for atom in poscar if atom.symbol == 'Si']

    dist_cutoff = 2.0
    idx_F_to_remove = [
        i for i in idx_F if np.min(D_len[i, idx_Si]) < dist_cutoff]
    print(f"Removed {len(idx_F_to_remove)} F atoms from the slab")

    poscar_copy = poscar.copy()
    while poscar_copy:
        poscar_copy.pop()

    for atom in poscar:
        if atom.index not in idx_F_to_remove:
            poscar_copy.append(atom)

    write('POSCAR_removed_slab', poscar_copy, format='vasp')


def remove_gas_HF(poscar, D_len):
    idx_F = [atom.index for atom in poscar if atom.symbol == 'F']
    idx_H = [atom.index for atom in poscar if atom.symbol == 'H']

    dist_cutoff = 1.1
    idx_F_to_remove = [
        i for i in idx_F if np.min(D_len[i, idx_H]) < dist_cutoff]
    print(f"Removed {len(idx_F_to_remove)} F atoms from the HF gas")

    poscar_copy = poscar.copy()
    while poscar_copy:
        poscar_copy.pop()

    for atom in poscar:
        if atom.index not in idx_F_to_remove:
            poscar_copy.append(atom)

    write('POSCAR_removed_HF', poscar_copy, format='vasp')


def main():
    poscar = read('POSCAR')
    pos = poscar.get_positions()

    _, D_len = get_distances(pos, cell=poscar.get_cell(), pbc=True)
    remove_slab_F(poscar, D_len)
    remove_gas_HF(poscar, D_len)


if __name__ == '__main__':
    main()
