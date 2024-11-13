import sys
from ase.io import read, write


def get_specorder(path_poscar, mol, pot_type):
    src = '/data2/andynn/LowTempEtch/03_gases'
    path_mol = f'{src}/{pot_type}/{mol}/POSCAR_relaxed'
    elements = []
    for path in [path_poscar, path_mol]:
        [elements.append(i) for i in read(path).get_chemical_symbols()
         if i not in elements]

    return elements


def main():
    if len(sys.argv) != 4:
        print('Usage: pos2lmp.py *POSCAR* *mol_type* *pot_type*')
        sys.exit(1)

    path_poscar, path_mol, pot_type = sys.argv[1:4]
    poscar = read(path_poscar)
    poscar.wrap()
    specorder = get_specorder(path_poscar, path_mol, pot_type)
    write('input.data', poscar, format='lammps-data', specorder=specorder)


main()
