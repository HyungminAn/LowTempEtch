import os
import yaml
import subprocess
from ase.io import read, write


class CellGenerator():
    def __init__(self, path_input_yaml):
        with open(path_input_yaml, 'r') as f:
            inputs = yaml.safe_load(f)

        self.mol_size = inputs['params']['mol_size']
        self.n_mol = inputs['params']['n_mol']
        self.tolerance = inputs['params']['tolerance']
        self.layer_additive = inputs['params']['layer_ADDI']
        self.layer_HF = inputs['params']['layer_HF']
        self.mol_size_HF = inputs['params']['mol_size_HF']
        self.n_HF = inputs['params']['n_HF']

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

        self.inc_size = 0

    def generate(self):
        self._write_packmol_input()
        result = self._run_packmol()
        while result != 0:
            self.inc_size += 0.5
            print(f'Increasing the layer thickness by {self.inc_size} A')
            self._write_packmol_input()
            result = self._run_packmol()
        self._merge()
        self._summarize()

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

        mol = read(self.path_mol)
        write("./mol.xyz", mol, format='xyz')
        self.path_mol = "./mol.xyz"

        poscar_HF = read(self.path_HF)
        write("./HF.xyz", poscar_HF, format='xyz')
        self.path_HF = "./HF.xyz"

        x_lim, y_lim, z_lim = _get_cell_params(self.path_poscar)
        pbc_padding = self.tolerance / 2

        params_dict = {
            'tolerance': self.tolerance,
            'filetype': 'xyz',
            'output_name': self.path_output,
            'x_lim': (pbc_padding, x_lim-pbc_padding),
            'y_lim': (pbc_padding, y_lim-pbc_padding),
            'z_lim': (z_lim+self.tolerance, z_lim+self.tolerance+layer_thickness),
        }
        with open(self.path_input, 'w') as f:
            line = f"tolerance {params_dict['tolerance']}\n"
            line += f"filetype {params_dict['filetype']}\n"
            line += f"output {params_dict['output_name']}\n"

            x_min, x_max = params_dict['x_lim']
            y_min, y_max = params_dict['y_lim']
            z_min, z_max = params_dict['z_lim']

            line += f"structure {self.path_mol}\n"
            line += f"  number {self.n_mol}\n"
            line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
            line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
            line += "end structure\n\n"

            line += f"structure {self.path_HF}\n"
            line += f"  number {self.n_HF}\n"
            line += f"  inside box {x_min:.1f} {y_min:.1f} {z_min:.1f}"
            line += f" {x_max:.1f} {y_max:.1f} {z_max:.1f}\n"
            line += "end structure\n"

            line += "seed -1\n"

            f.write(line)

    def _run_packmol(self):
        '''
        Run PACKMOL, using the input file
        '''
        command = f"{self.path_packmol} < {self.path_input} 2>&1 >> {self.path_log}"
        result = subprocess.run(
            command, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode

        return result

    def _merge(self):
        poscar = read(self.path_poscar)
        result = read(self.path_output)

        for atom in result:
            poscar.append(atom)

        cell = poscar.get_cell()
        cell[2][2] += 10.0
        poscar.set_cell(cell)

        write("POSCAR_merged", poscar, format='vasp', sort=True)

    def _summarize(self):
        with open("result.yaml", "w") as f:
            yaml.dump(self.__dict__, f)

def _get_cell_params(path_poscar):
    '''
    Get the cell parameters from the POSCAR file
    '''
    atoms = read(path_poscar)
    cell = atoms.get_cell()
    x_lim = cell[0, 0]
    y_lim = cell[1, 1]
    z_lim = max([atom.position[2] for atom in atoms])
    return x_lim, y_lim, z_lim
