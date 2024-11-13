import sys
import yaml
from ase.io import read


def get_cell_params(path_poscar):
    atoms = read(path_poscar)
    cell = atoms.get_cell()
    x_lim = cell[0, 0]
    y_lim = cell[1, 1]
    z_lim = max([atom.position[2] for atom in atoms])
    return x_lim, y_lim, z_lim


def main():
    if len(sys.argv) != 2:
        print("Usage: python write_inp.py input.yaml")
        sys.exit(1)

    with open ('input.yaml', 'r') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    mol_size = inputs['mol_size']
    tolerance = inputs['tolerance']
    path_poscar = inputs['path_poscar']
    path_mol = inputs['path_mol']
    n_mol = inputs['n_mol']

    x_lim, y_lim, z_lim = get_cell_params(path_poscar)
    pbc_padding = tolerance / 2

    params_dict = {
        'tolerance': tolerance,
        'filetype': 'xyz',
        'output_name': 'my_output.xyz',
        'x_lim': (pbc_padding, x_lim-pbc_padding),
        'y_lim': (pbc_padding, y_lim-pbc_padding),
        'z_lim': (z_lim+tolerance, z_lim+tolerance+mol_size),
    }
    with open('my_input.inp', 'w') as f:
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


if __name__ == '__main__':
    main()
