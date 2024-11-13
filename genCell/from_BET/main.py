import os
import sys
import yaml
import subprocess
import shutil
from itertools import combinations_with_replacement as H

import numpy as np

from ase.io import read, write
from ase.data import covalent_radii, atomic_numbers
from ase.constraints import FixAtoms


def pos2xyz(path_poscar, **inputs):
    '''
    Convert POSCAR to XYZ
    '''
    path_slab_xyz = inputs['path']['path_slab_xyz']
    atoms = read(path_poscar)
    write(path_slab_xyz, atoms, format='xyz')


def get_cell_params(path_poscar):
    '''
    Get the cell parameters from the POSCAR file
    '''
    atoms = read(path_poscar)
    cell = atoms.get_cell()
    x_lim = cell[0, 0]
    y_lim = cell[1, 1]
    z_lim = cell[2, 2]
    z_atom_min = min([atom.position[2] for atom in atoms])
    z_atom_max = max([atom.position[2] for atom in atoms])
    return x_lim, y_lim, z_lim, z_atom_min, z_atom_max


def write_packmol_input(**inputs):
    '''
    Write the input file for packmol
    '''
    mol_size = inputs['params']['mol_size']
    tolerance = inputs['params']['tolerance']
    n_mol = inputs['params']['n_mol']

    path_poscar = inputs['path']['path_poscar']
    path_mol = inputs['path']['path_mol']
    path_input = inputs['path']['path_input']
    path_output = inputs['path']['path_output']
    path_slab_xyz = inputs['path']['path_slab_xyz']

    x_lim, y_lim, z_lim, z_atom_min, z_atom_max = get_cell_params(path_poscar)
    x_shift = x_lim / 2
    y_shift = y_lim / 2
    z_shift = (z_atom_max - z_atom_min) / 2
    pos2xyz(path_poscar, **inputs)

    params_dict = {
        'tolerance': tolerance,
        'filetype': 'xyz',
        'output_name': path_output,
        'x_lim': (0, x_lim),
        'y_lim': (0, y_lim),
        'z_lim': (z_atom_max, z_atom_max+mol_size+tolerance*2),
    }

    with open(path_input, 'w') as f:
        line = f"tolerance {params_dict['tolerance']}\n"
        line += f"filetype {params_dict['filetype']}\n"
        line += f"pbc {x_lim} {y_lim} {z_lim}\n"
        line += f"output {params_dict['output_name']}\n"
        line += f"seed {np.random.randint(1000000)}\n"
        line += "randominitialpoint\n"

        line += f"structure {path_slab_xyz}\n"
        line += "  number 1\n"
        line += "  center\n"
        line += f"  fixed {x_shift} {y_shift} {z_shift} .0 .0 .0\n"
        line += "end structure\n"

        line += f"structure {path_mol}\n"
        line += f"  number {n_mol}\n"
        x_min, x_max = params_dict['x_lim']
        y_min, y_max = params_dict['y_lim']
        z_min, z_max = params_dict['z_lim']
        line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
        line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
        line += "end structure\n"
        f.write(line)


def run_packmol(**inputs):
    '''
    Run PACKMOL, using the input file
    '''
    path_packmol = inputs['path']['path_packmol']
    path_input = inputs['path']['path_input']
    path_log = inputs['path']['path_log']

    command = f"{path_packmol} < {path_input} 2>&1 >> {path_log}"
    result = subprocess.run(
        command, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode
    return result


def write_output(to_merge, **inputs):
    path_result = inputs['path']['path_output']
    result = read(path_result)

    path_poscar = inputs['path']['path_poscar']
    poscar = read(path_poscar)
    cell = poscar.get_cell()
    if to_merge:
        for atom in result:
            poscar.append(atom)
    else:
        poscar = result
    poscar.set_cell(cell)

    z_fix = 4.0
    c = FixAtoms(indices=[
        atom.index for atom in poscar if atom.position[2] < z_fix])
    poscar.set_constraint(c)

    write("POSCAR_merged", poscar, format='vasp', sort=True)


def get_tolerance(mol_structure):
    '''
    Get the tolerance for the molecule
    '''
    elem = list(set(mol_structure.get_chemical_symbols()))
    r_cov = [covalent_radii[atomic_numbers[e]] for e in elem]
    r_cov_max = max([i+j for i, j in H(r_cov, 2)])
    return float(r_cov_max)


def determine_layer_thickness(
        init_thickness=0.5, ratio_1=1.2, ratio_2=1.5, **inputs):
    '''
    Get the thickness of the layer for the molecule
    '''
    if inputs['params']['n_mol'] is not None:
        inputs['params']['test_n_mol'] = False

    inputs['params']['n_mol'] = 1
    path_mol = inputs['path']['path_mol']

    mol = read(path_mol)
    n_atoms = len(mol)
    inputs['params']['tolerance'] = get_tolerance(mol)

    if n_atoms < 3:  # diatomic molecule
        pos = mol.get_positions()
        bond_length = np.linalg.norm(pos[0] - pos[1])
        inputs['params']['mol_size'] = float(bond_length * ratio_1)
        write_packmol_input(**inputs)
        result = run_packmol(**inputs)
        return

    inputs['params']['mol_size'] = init_thickness
    write_packmol_input(**inputs)
    result = run_packmol(**inputs)
    while result != 0:
        inputs['params']['mol_size'] += 0.1
        write_packmol_input(**inputs)
        result = run_packmol(**inputs)
    inputs['params']['mol_size'] *= ratio_2


def determine_number_of_molecules(**inputs):
    if not inputs['params']['test_n_mol']:
        return

    write_packmol_input(**inputs)
    result = run_packmol(**inputs)
    is_success = result == 0
    while is_success:
        inputs['params']['n_mol'] += 1
        write_packmol_input(**inputs)
        result = run_packmol(**inputs)
        is_success = result == 0
    inputs['params']['n_mol'] -= 1
    write_packmol_input(**inputs)
    result = run_packmol(**inputs)


def adjust_thickness_manually(**inputs):
    mol_name = inputs['params']['mol_name']

    adjust_ratio_dict = {
        'BrF5': 0.9,
        'CF3I': 1.2,
        'IBr': 0.7,
        'IF5': 1.3,
        'NH4F': 0.7,
        'WF6': 1.5,
        'TaF5': 1.6,
    }

    if mol_name in adjust_ratio_dict:
        inputs['params']['mol_size'] *= adjust_ratio_dict[mol_name]


def repeat_run(**inputs):
    '''
    Repeat packmol run, for the given number of repetitions
    '''
    n_repeats = inputs['params']['repeat']
    to_merge = inputs['params']['to_merge']
    path_input = inputs['path']['path_input']

    for i in range(n_repeats):
        dst = f"{i}"
        is_done = False

        while not is_done:
            write_packmol_input(**inputs)
            is_done = run_packmol(**inputs) == 0
            if not is_done:
                print(f"{i} Failed! Retrying...")

        write_output(to_merge, **inputs)

        os.makedirs(dst, exist_ok=True)
        shutil.move('POSCAR_merged', f'{dst}/POSCAR_merged')
        shutil.move('LOG', f'{dst}/LOG')
        shutil.move(path_input, f'{dst}/{path_input}')
        print(f"{i} Done!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py input.yaml")
        sys.exit(1)
    path_input_yaml = sys.argv[1]
    with open(path_input_yaml, 'r') as f:
        inputs = yaml.safe_load(f)

    determine_layer_thickness(**inputs)
    adjust_thickness_manually(**inputs)
    determine_number_of_molecules(**inputs)
    repeat_run(**inputs)

    with open('result.yaml', 'w') as f:
        yaml.dump(inputs, f)


if __name__ == "__main__":
    main()
