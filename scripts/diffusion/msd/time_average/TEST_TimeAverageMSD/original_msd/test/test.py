from ase.io import read, write


def main():
    dump = read('../dump.lammps', index=-1)
    write('POSCAR', dump, format='vasp')


if __name__ == '__main__':
    main()
