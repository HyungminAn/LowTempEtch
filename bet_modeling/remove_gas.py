import sys

import numpy as np

from ase.constraints import FixAtoms
from ase.io import read, write
from ase.geometry import get_distances
from ase.data import atomic_numbers, covalent_radii


def get_mol_data():
    center_atom_dict = {
        'AsF5': [('As', 1)],
        'BF3': [('B', 1)],
        'BiF5': [('Bi', 1)],
        'BrF5': [('Br', 1)],
        'C2I2F4': [('C', 2)],
        'C2IF5': [('C', 2)],
        'CF3I': [('C', 1)],
        'COF2': [('C', 1)],
        'COS': [('C', 1)],
        'CS2': [('C', 1)],
        'ClF5': [('Cl', 1)],
        'CoCl2': [('Co', 1)],
        'HF': [('H', 1)],
        'HI': [('H', 1)],
        'HfCl4': [('Hf', 1)],
        'IBr': [('I', 1)],
        'IF5': [('I', 1)],
        'IF7': [('I', 1)],
        'NH4F': [('N', 1)],
        'NbF5': [('Nb', 1)],
        'PF3': [('P', 1)],
        'PF5': [('P', 1)],
        'SO2': [('S', 1)],
        'TaCl4': [('Ta', 1)],
        'TaF5': [('Ta', 1)],
        'TiCl4': [('Ti', 1)],
        'WF6': [('W', 1)],
        'XeF2': [('Xe', 1)],
    }

    surrond_atom_dict = {
        'AsF5': [('F', 5)],
        'BF3': [('F', 3)],
        'BiF5': [('F', 5)],
        'BrF5': [('F', 5)],
        'C2I2F4': [('I', 2), ('F', 4)],
        'C2IF5': [('I', 1), ('F', 5)],
        'CF3I': [('F', 3), ('I', 1)],
        'COF2': [('O', 1), ('F', 2)],
        'COS': [('O', 1), ('S', 1)],
        'CS2': [('S', 2)],
        'ClF5': [('F', 5)],
        'CoCl2': [('Cl', 2)],
        'HF': [('F', 1)],
        'HI': [('I', 1)],
        'HfCl4': [('Cl', 4)],
        'IBr': [('Br', 1)],
        'IF5': [('F', 5)],
        'IF7': [('F', 7)],
        'NH4F': [('H', 4), ('F', 1)],
        'NbF5': [('F', 5)],
        'PF3': [('F', 3)],
        'PF5': [('F', 5)],
        'SO2': [('O', 2)],
        'TaCl4': [('Cl', 4)],
        'TaF5': [('F', 5)],
        'TiCl4': [('Cl', 4)],
        'WF6': [('F', 6)],
        'XeF2': [('F', 2)],
    }

    return center_atom_dict, surrond_atom_dict


def make_bond_length_dict(center_atom_type, surr_atom_types, factor=1.4):
    bond_length_dict = {}
    for surr_atom_type, _ in surr_atom_types:
        r = covalent_radii[atomic_numbers[center_atom_type]] + \
            covalent_radii[atomic_numbers[surr_atom_type]]
        bond_length_dict[(center_atom_type, surr_atom_type)] = r * factor
        bond_length_dict[(surr_atom_type, center_atom_type)] = r * factor

    r = covalent_radii[atomic_numbers[center_atom_type]] + \
        covalent_radii[atomic_numbers[center_atom_type]]
    bond_length_dict[(center_atom_type, center_atom_type)] = r * factor

    return bond_length_dict


def get_center_atom_indices(poscar, center_atom_dict, gas_name, D_len):
    '''
    Only supports up to two centers, with one element type.

    For more than two centers,
    graph-based algorithm is needed.
    '''
    atom_type, count = center_atom_dict[gas_name][0]
    pos = poscar.get_positions()

    idx_list = [
        (atom.index, pos[atom.index, 2]) for atom in poscar
        if atom.symbol == atom_type
    ]
    idx_top = sorted(idx_list, key=lambda x: x[1])[-1][0]
    if count == 1:
        idx_top = np.array([idx_top])
        return idx_top
    elif count == 2:
        idx_list = np.array([i[0] for i in idx_list])
        idx_NN = np.argsort(D_len[idx_top, idx_list])[1]
        idx_2ndCenter = np.array(idx_list)[idx_NN]
        return np.array([idx_top, idx_2ndCenter])


def get_surr_atom_indices(
        poscar, center_atom_type, idx_center_atoms, surr_atom_types,
        D_len, bond_length_dict, gas_name=None, cutoff_inc=0.1):

    result = []
    for surr_atom_type, surr_atom_count in surr_atom_types:
        idx_list_surr = []
        cutoff = bond_length_dict[(surr_atom_type, center_atom_type)]

        while len(idx_list_surr) < surr_atom_count:
            for idx_center_atom in idx_center_atoms:
                idx_surr_atoms = [
                    (atom.index,
                     D_len[idx_center_atom, atom.index],
                     atom.position[2])
                    for atom in poscar if atom.symbol == surr_atom_type
                    and D_len[idx_center_atom, atom.index] < cutoff
                ]
                idx_list_surr += idx_surr_atoms
            cutoff += cutoff_inc
        if len(idx_list_surr) > surr_atom_count:
            print(f"Warning {gas_name}: Too many surrounding atoms")
            idx_list_surr = sorted(idx_list_surr, key=lambda x: x[1])
            idx_list_surr = idx_list_surr[:surr_atom_count]
        result += [i[0] for i in idx_list_surr]
    return result


def get_idx_to_remove(poscar, gas_name):
    center_atom_dict, surrond_atom_dict = get_mol_data()

    center_atom_type = center_atom_dict[gas_name][0][0]
    surr_atom_types = surrond_atom_dict[gas_name]
    bond_length_dict = make_bond_length_dict(center_atom_type, surr_atom_types)

    pos = poscar.get_positions()
    _, D_len = get_distances(pos, pbc=True, cell=poscar.get_cell())
    idx_center_atoms = get_center_atom_indices(
        poscar, center_atom_dict, gas_name, D_len)

    idx_surr_atoms = get_surr_atom_indices(
        poscar, center_atom_type, idx_center_atoms, surr_atom_types,
        D_len, bond_length_dict, gas_name=gas_name)

    idx_to_remove = [
        int(i) for i in idx_surr_atoms] + [
        int(i) for i in idx_center_atoms]

    return idx_to_remove


def write_poscar(poscar, idx_to_remove, dst, fix_height=None):
    poscar_removed = poscar.copy()
    while poscar_removed:
        poscar_removed.pop()

    for i, atom in enumerate(poscar):
        if i in idx_to_remove:
            continue
        poscar_removed.append(atom)

    c = FixAtoms(
        indices=[atom.index for atom in poscar_removed
                 if atom.position[2] < fix_height]
        )

    poscar_removed.set_constraint(c)
    write(dst, poscar_removed, format='vasp')


def main():
    if len(sys.argv) != 4:
        print("Usage: python remove_gas.py POSCAR *gas_name* POSCAR_out")
        sys.exit(1)

    fix_height = 4.0

    path_poscar, gas_name, dst = sys.argv[1:4]
    poscar = read(path_poscar)

    idx_to_remove = get_idx_to_remove(poscar, gas_name)
    write_poscar(poscar, idx_to_remove, dst, fix_height=fix_height)


if __name__ == '__main__':
    main()
