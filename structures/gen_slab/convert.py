import sys

import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms
from ase import Atom


def slice_cell(poscar, **params_dict):
    poscar_new = poscar.copy()
    while poscar_new:
        poscar_new.pop()

    slice_lower_limit = params_dict.get('slice_lower_limit')
    slice_upper_limit = params_dict.get('slice_upper_limit')

    for atom in poscar:
        cond = atom.position[2] < slice_lower_limit\
               or atom.position[2] > slice_upper_limit
        if cond:
            continue
        poscar_new.append(atom)

    return poscar_new


def add_vacuum(poscar, **params_dict):
    cell = poscar.get_cell()
    vacuum_length = params_dict.get('vacuum_length')
    cell[2][2] += vacuum_length
    poscar.set_cell(cell)


def add_hydrogen(poscar, **params_dict):
    h_top = params_dict.get('h_top')
    h_bot = params_dict.get('h_bot')
    h_shift = params_dict.get('h_shift')

    idx_top_O = [
        atom.index for atom in poscar
        if atom.symbol == 'O' and atom.position[2] > h_top]
    idx_bot_O = [
        atom.index for atom in poscar
        if atom.symbol == 'O' and atom.position[2] < h_bot]
    print(idx_top_O)
    print(idx_bot_O)

    for idx in idx_top_O:
        x, y, z = poscar[idx].position
        z += h_shift
        poscar.append(Atom('H', (x, y, z)))
    for idx in idx_bot_O:
        x, y, z = poscar[idx].position
        z -= h_shift
        poscar.append(Atom('H', (x, y, z)))


def shift_and_fix(poscar, **params_dict):
    min_shift = params_dict.get('min_shift')
    pos = poscar.get_positions()
    z_min = np.min(pos[:, 2])
    pos[:, 2] -= (z_min - min_shift)
    poscar.set_positions(pos, apply_constraint=False)

    fix_h = params_dict.get('fix_height')
    c = FixAtoms([atom.index for atom in poscar if atom.position[2] < fix_h])
    poscar.set_constraint(c)


def main():
    params_dict = {
        'slice_lower_limit': -100.0,  # Angstrom
        'slice_upper_limit': 100.0,  # Angstrom
        'vacuum_length': 15.0,  # Angstrom
        'h_top': 10.0,  # height of top O atom to add H
        'h_bot': 0.5,  # height of bottom O atom to add H
        'h_shift': 0.5,  # shift of H atom from O atom
        'min_shift': 0.5,  # minimum shift of the structure in z-direction
        'fix_height': 4.5,  # height of the fixed atoms
    }

    if len(sys.argv) < 2:
        print('Usage: python convert.py POSCAR')
        sys.exit(1)

    path_poscar = sys.argv[1]
    poscar = read(path_poscar)
    # poscar = slice_cell(poscar, **params_dict)
    add_vacuum(poscar, **params_dict)
    # add_hydrogen(poscar, **params_dict)
    shift_and_fix(poscar, **params_dict)

    write('POSCAR', poscar, format='vasp')


main()
