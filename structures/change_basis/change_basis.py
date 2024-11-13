from ase.io import read, write


def main():
    poscar = read('POSCAR_211')
    cell = poscar.get_cell()
    x = [cell[0][0], 0, 0]
    y = [0, abs(cell[0][1]), 0]
    cell[0, :] = x
    cell[1, :] = y
    poscar.set_cell(cell)
    poscar.wrap()
    write('POSCAR_new', poscar, format='vasp')


if __name__ == '__main__':
    main()
