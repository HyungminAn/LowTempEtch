import sys

from ase.io import read, write


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge.py POSCAR result.xyz")
        sys.exit(1)

    path_poscar, path_result = sys.argv[1:3]
    poscar = read(path_poscar)
    result = read(path_result)

    for atom in result:
        poscar.append(atom)

    write("POSCAR_merged", poscar, format='vasp', sort=True)


if __name__ == "__main__":
    main()
