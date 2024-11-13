import sys
import yaml
import subprocess

import numpy as np

from ase.io import read


def get_cell_params(path_poscar):
    '''
    Get the cell parameters from the POSCAR file
    '''
    atoms = read(path_poscar)
    cell = atoms.get_cell()
    x_lim = cell[0, 0]
    y_lim = cell[1, 1]
    z_lim = max([atom.position[2] for atom in atoms])
    return x_lim, y_lim, z_lim


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

    x_lim, y_lim, z_lim = get_cell_params(path_poscar)
    pbc_padding = tolerance / 2

    params_dict = {
        'tolerance': tolerance,
        'filetype': 'xyz',
        'output_name': path_output,
        'x_lim': (pbc_padding, x_lim-pbc_padding),
        'y_lim': (pbc_padding, y_lim-pbc_padding),
        'z_lim': (z_lim+tolerance, z_lim+tolerance+mol_size),
    }
    with open(path_input, 'w') as f:
        line = f"tolerance {params_dict['tolerance']}\n"
        line += f"filetype {params_dict['filetype']}\n"
        line += f"output {params_dict['output_name']}\n"

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


def merge(**inputs):
    path_poscar = inputs['path']['path_poscar']
    path_result = inputs['path']['path_output']
    poscar = read(path_poscar)
    result = read(path_result)

    for atom in result:
        poscar.append(atom)

    write("POSCAR_merged", poscar, format='vasp', sort=True)


def determine_layer_thickness(
        init_thickness=0.5, ratio_1=1.2, ratio_2=1.5, **inputs):
    '''
    Get the thickness of the layer for the molecule
    '''
    inputs['params']['n_mol'] = 1
    path_mol = inputs['path']['path_mol']

    mol = read(path_mol)
    n_atoms = len(mol)

    if n_atoms < 3:  # diatomic molecule
        pos = mol.get_positions()
        bond_length = np.linalg.norm(pos[0] - pos[1])
        inputs['params']['mol_size'] = bond_length * ratio_1
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
    result = run_packmol(**inputs)
    while result == 0:
        inputs['params']['n_mol'] += 1
        write_packmol_input(**inputs)
        result = run_packmol(**inputs)
    inputs['params']['n_mol'] -= 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py input.yaml")
        sys.exit(1)
    path_input_yaml = sys.argv[1]
    with open(path_input_yaml, 'r') as f:
        inputs = yaml.safe_load(f)

    determine_layer_thickness(**inputs)
    determine_number_of_molecules(**inputs)
    run_packmol(**inputs)
    merge(**inputs)


if __name__ == "__main__":
    main()
