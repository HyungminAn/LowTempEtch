import os
import sys
import yaml
import subprocess

from ase.io import read, write


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
    layer_ADDI = inputs['params']['layer_ADDI']
    n_mol = round(n_mol * layer_ADDI)

    mol_size_HF = inputs['params']['mol_size_HF']
    n_HF = inputs['params']['n_HF']
    layer_HF = inputs['params']['layer_HF']
    n_HF = round(n_HF * layer_HF)

    inc_size = inputs['params']['inc_size']

    layer_thickness = mol_size * layer_ADDI + mol_size_HF * layer_HF + inc_size

    path_poscar = inputs['path']['path_poscar']
    if not os.path.exists(path_poscar):
        raise FileNotFoundError(f"File not found: {path_poscar}")
    path_mol = inputs['path']['path_mol']
    if not os.path.exists(path_mol):
        raise FileNotFoundError(f"File not found: {path_mol}")
    mol = read(path_mol)
    write("./mol.xyz", mol, format='xyz')
    path_mol = "./mol.xyz"

    path_input = inputs['path']['path_input']
    path_output = inputs['path']['path_output']

    path_HF = inputs['path']['path_HF']
    poscar_HF = read(path_HF)
    write("./HF.xyz", poscar_HF, format='xyz')
    path_HF = "./HF.xyz"

    x_lim, y_lim, z_lim = get_cell_params(path_poscar)
    pbc_padding = tolerance / 2

    params_dict = {
        'tolerance': tolerance,
        'filetype': 'xyz',
        'output_name': path_output,
        'x_lim': (pbc_padding, x_lim-pbc_padding),
        'y_lim': (pbc_padding, y_lim-pbc_padding),
        'z_lim': (z_lim+tolerance, z_lim+tolerance+layer_thickness),
    }
    with open(path_input, 'w') as f:
        line = f"tolerance {params_dict['tolerance']}\n"
        line += f"filetype {params_dict['filetype']}\n"
        line += f"output {params_dict['output_name']}\n"

        x_min, x_max = params_dict['x_lim']
        y_min, y_max = params_dict['y_lim']
        z_min, z_max = params_dict['z_lim']

        line += f"structure {path_mol}\n"
        line += f"  number {n_mol}\n"
        line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
        line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
        line += "end structure\n\n"

        line += f"structure {path_HF}\n"
        line += f"  number {n_HF}\n"
        line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
        line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
        line += "end structure\n"

        line += "seed -1\n"

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

    cell = poscar.get_cell()
    cell[2][2] += 10.0
    poscar.set_cell(cell)

    write("POSCAR_merged", poscar, format='vasp', sort=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py input.yaml")
        sys.exit(1)
    path_input_yaml = sys.argv[1]
    with open(path_input_yaml, 'r') as f:
        inputs = yaml.safe_load(f)

    inputs['params']['inc_size'] = 0
    write_packmol_input(**inputs)
    result = run_packmol(**inputs)
    while result != 0:
        inputs['params']['inc_size'] += 0.5
        print(f'Increasing the layer thickness by {inputs["params"]["inc_size"]}')
        write_packmol_input(**inputs)
        result = run_packmol(**inputs)
    merge(**inputs)

    with open("result.yaml", 'w') as f:
        yaml.dump(inputs, f)


if __name__ == "__main__":
    main()
