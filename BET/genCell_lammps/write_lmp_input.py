import sys
from ase.data import atomic_numbers, atomic_masses
from ase.io import read


def get_elem_order(*path_poscars):
    elements = []
    for path_poscar in path_poscars:
        poscar = read(path_poscar)
        [elements.append(i) for i in poscar.get_chemical_symbols()
         if i not in elements]
    return elements


def write_mol(path_gas, elements, mol):
    image = read(path_gas)
    pos = image.get_positions()
    n_atoms = len(image)
    type_list = [elements.index(atom.symbol) + 1 for atom in image]

    with open(f'mol_{mol}', 'w') as f:
        w = f.write

        w(f"# {mol}\n\n")
        w(f"{n_atoms} atoms\n\n")

        w("Coords\n\n")
        for idx, xyz in enumerate(pos):
            w(f"{idx+1} {xyz[0]} {xyz[1]} {xyz[2]}\n")
        w("\n")

        w("Types\n\n")
        for idx, elem_type in enumerate(type_list):
            w(f"{idx+1} {elem_type}\n")
        w("\n")

        w("Masses\n\n")
        for idx, atom in enumerate(image):
            mass = atomic_masses[atomic_numbers[atom.symbol]]
            w(f"{idx+1} {mass}\n")


def write_lmp_input(path_slab, path_gas, mol_name):
    line_header = '''
#LAMMPS
#########################################################
#                      Basic Input                      #
#########################################################
units           metal     # K, bar, ps, A
box             tilt large
boundary        p p f
newton          on

read_data       input.data
'''

    elements = get_elem_order(path_slab, path_gas)
    # Write HF molecule data file
    write_mol(path_gas, elements, mol_name)

    line_mass = ''
    for elem in elements:
        mass = atomic_masses[atomic_numbers[elem]]
        line_mass += f'variable m_{elem} equal {mass}\n'
    line_mass += '\n'
    for idx, elem in enumerate(elements):
        line_mass += f'mass  {idx+1}  ${{m_{elem}}}\n'

    element_list = ' '.join(elements)
    line_elem = f'variable element_list string "{element_list}"\n'
    line_elem += '\n'

    line_footer = f'''
#########################################################
#                   Insert molecules                    #
#########################################################
molecule    my_mol      "mol_{mol_name}"
region      rDepo       block   EDGE EDGE  EDGE EDGE  EDGE EDGE
fix         fDepo       all     deposit     1 0 1000000 ${{SEEDS}} &
                        region  rDepo    mol my_mol &
                        local   2.0 3.0 2.0
run         1

#########################################################
#                     Finalize                          #
#########################################################
write_data      FINAL.coo
'''

    with open('lammps.in', 'w') as f:
        f.write(line_header)
        f.write(line_mass)
        f.write(line_elem)
        f.write(line_footer)


def main():
    if len(sys.argv) < 3:
        print('Usage: python -.py *path_slab* *gas_name* *sevenn_type*')
        sys.exit(1)

    path_slab, mol, pot_type = sys.argv[1:4]
    path_src = '/data2/andynn/LowTempEtch/03_gases'
    path_gas = f'{path_src}/{pot_type}/{mol}/POSCAR_relaxed'

    write_lmp_input(path_slab, path_gas, mol)


main()
