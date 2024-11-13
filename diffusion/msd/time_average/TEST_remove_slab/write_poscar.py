from ase.io import read, write


def main():
    atoms = read('../../../03_reproduce_TEST/IF5/originalCell/1/dump.lammps')

    write('POSCAR', atoms, format='vasp')


main()
