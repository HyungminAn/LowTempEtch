import os
import sys
import yaml
import subprocess

import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.geometry import get_distances


class CellGenerator():
    def __init__(self, path_input_yaml):
        with open(path_input_yaml, 'r') as f:
            inputs = yaml.safe_load(f)

        self.tolerance = inputs['params']['tolerance']
        self.layer_additive = inputs['params']['layer_ADDI']
        self.layer_HF = inputs['params']['layer_HF']

        self.path_mol = inputs['path']['path_mol']
        self.path_poscar = inputs['path']['path_poscar']
        self.path_HF = inputs['path']['path_HF']
        self.path_packmol = inputs['path']['path_packmol']
        check_list = [
            self.path_poscar,
            self.path_mol,
            self.path_HF,
            self.path_packmol
            ]
        for file in check_list:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")

        self.path_input = inputs['path']['path_input']
        self.path_output = inputs['path']['path_output']
        self.path_log = inputs['path']['path_log']

        self.mol_size = _get_mol_size(self.path_mol)
        self.mol_size_HF = _get_mol_size(self.path_HF)

        self.n_mol = self._get_mol_number(self.path_mol, self.mol_size)
        self.n_HF = self._get_mol_number(self.path_HF, self.mol_size_HF)

        self.inc_size = 0
        self.dst = 'POSCAR'

    def generate(self):
        self._write_packmol_input()
        result = _run_packmol(self.path_packmol, self.path_input, self.path_log)
        while result != 0:
            self.inc_size += 0.5
            print(f'Increasing the layer thickness by {self.inc_size} A')
            self._write_packmol_input()
            result = _run_packmol(self.path_packmol, self.path_input, self.path_log)
        with open("result.yaml", "w") as f:
            yaml.dump(self.__dict__, f)
        self._write()

    def _write_packmol_input(self):
        '''
        Write the input file for packmol
        '''
        self.n_mol = round(self.n_mol * self.layer_additive)
        self.n_HF = round(self.n_HF * self.layer_HF)

        layer_thickness =\
              self.mol_size    * self.layer_additive\
            + self.mol_size_HF * self.layer_HF\
            + self.inc_size

        x_lim, y_lim, z_lim, z_atom_min, z_atom_max = _get_cell_params(self.path_poscar)
        x_shift = x_lim / 2
        y_shift = y_lim / 2
        z_shift = (z_atom_max - z_atom_min) / 2

        slab = read(self.path_poscar)
        write("./slab.xyz", slab, format='xyz')
        self.path_poscar_xyz = "./slab.xyz"

        mol = read(self.path_mol)
        write("./mol.xyz", mol, format='xyz')
        self.path_mol_xyz = "./mol.xyz"

        poscar_HF = read(self.path_HF)
        write("./HF.xyz", poscar_HF, format='xyz')
        self.path_HF_xyz = "./HF.xyz"

        params_dict = {
            'tolerance': self.tolerance,
            'filetype': 'xyz',
            'output_name': self.path_output,
            'x_lim': (0, x_lim),
            'y_lim': (0, y_lim),
            'z_lim': (z_atom_max, z_atom_max+layer_thickness+self.tolerance*2),
        }
        with open(self.path_input, 'w') as f:
            line = f"tolerance {params_dict['tolerance']}\n"
            line += f"filetype {params_dict['filetype']}\n"
            line += f"pbc {x_lim} {y_lim} {z_lim}\n"
            line += f"output {params_dict['output_name']}\n"
            line += "randominitialpoint\n"

            line += f"structure {self.path_poscar_xyz}\n"
            line += "  number 1\n"
            line += "  center\n"
            line += f"  fixed {x_shift} {y_shift} {z_shift} .0 .0 .0\n"
            line += "end structure\n"

            x_min, x_max = params_dict['x_lim']
            y_min, y_max = params_dict['y_lim']
            z_min, z_max = params_dict['z_lim']

            line += f"structure {self.path_mol_xyz}\n"
            line += f"  number {self.n_mol}\n"
            line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
            line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
            line += "end structure\n\n"

            line += f"structure {self.path_HF_xyz}\n"
            line += f"  number {self.n_HF}\n"
            line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
            line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
            line += "end structure\n"

            line += "seed -1\n"

            f.write(line)

    def _get_mol_number(self, path_mol, mol_size):
        mol_number = 1
        self._write_packmol_input_mol(path_mol, mol_size, mol_number)
        result = _run_packmol(self.path_packmol, self.path_input, self.path_log)
        while result == 0:
            mol_number += 1
            # print(f'Increasing mol_number by {mol_number}')
            self._write_packmol_input_mol(path_mol, mol_size, mol_number)
            result = _run_packmol(self.path_packmol, self.path_input, self.path_log)
        print(f'Number of molecules: {mol_number}')
        return mol_number

    def _write_packmol_input_mol(self, path_mol, mol_size, n_mol):
        x_lim, y_lim, z_lim, *_ = _get_cell_params(self.path_poscar)

        mol = read(path_mol)
        write("./mol.xyz", mol, format='xyz')
        path_mol_xyz = "./mol.xyz"

        params_dict = {
            'tolerance': self.tolerance,
            'filetype': 'xyz',
            'output_name': self.path_output,
            'x_lim': (0, x_lim),
            'y_lim': (0, y_lim),
            'z_lim': (0, mol_size),
        }
        with open(self.path_input, 'w') as f:
            line = f"tolerance {params_dict['tolerance']}\n"
            line += f"filetype {params_dict['filetype']}\n"
            line += f"pbc {x_lim} {y_lim} {z_lim}\n"
            line += f"output {params_dict['output_name']}\n"
            line += "randominitialpoint\n"

            x_min, x_max = params_dict['x_lim']
            y_min, y_max = params_dict['y_lim']
            z_min, z_max = params_dict['z_lim']

            line += f"structure {path_mol_xyz}\n"
            line += f"  number {n_mol}\n"
            line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
            line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
            line += "end structure\n\n"

            line += "seed -1\n"

            f.write(line)

    def _write(self):
        result = read(self.path_output)
        z_min = result.get_positions()[:, 2].min()
        cell = read(self.path_poscar).get_cell()

        if z_min < 0:
            z_shift = -z_min
            pos = result.get_positions()
            pos[:, 2] += z_shift
            result.set_positions(pos)

        z_max = result.get_positions()[:, 2].max()
        z_shift = cell[2, 2] - z_max
        if 0 < z_shift < 1:
            pos = result.get_positions()
            pos[:, 2] += z_shift
            result.set_positions(pos)

        z_max = result.get_positions()[:, 2].max()
        if z_max + 20 > cell[2, 2]:
            cell[2, 2] = z_max + 20
        result.set_cell(cell)
        result.wrap()

        z_fix = 4.0
        c = FixAtoms(indices=[
            atom.index for atom in result if atom.position[2] < z_fix])
        result.set_constraint(c)
        print(f"Writing the final structure to {self.dst}")
        write(self.dst, result, format='vasp', sort=True)


def _get_cell_params(path_poscar):
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


def _get_mol_size(path_poscar):
    atoms = read(path_poscar)
    _, D_len = get_distances(atoms.get_positions(), cell=atoms.get_cell(), pbc=True)
    return float(np.max(D_len))


def _run_packmol(path_packmol, path_input, path_log):
    '''
    Run PACKMOL, using the input file
    '''
    command = f"{path_packmol} < {path_input} 2>&1 >> {path_log}"
    result = subprocess.run(
        command, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode

    return result


if __name__ == "__main__":
    path_input_yaml = sys.argv[1]
    gen = CellGenerator(path_input_yaml)
    gen.generate()
