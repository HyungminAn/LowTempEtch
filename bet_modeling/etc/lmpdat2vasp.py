from ase.io import read, write
from ase.constraints import FixAtoms
import sys


def main(path_data, *args, **kwargs):
    '''
    Read lammps data file and convert to VASP POSCAR format
    '''
    data = read(path_data, format='lammps-data', atom_style='atomic')

    pos = data.get_positions()
    data.set_positions(pos, apply_constraint=False)
    data.wrap()

    c = FixAtoms(indices=[
            atom.index for atom in data
            if atom.position[2] <= 4.0
    ])
    data.set_constraint(c)
    write('POSCAR', data, format='vasp', direct=False, sort=True)

    return


main(sys.argv[1])
