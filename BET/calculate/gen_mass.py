import re
from ase.data import atomic_masses, atomic_names, atomic_numbers


def main():
    with open('mol_list.dat', 'r') as f:
        mol_list = f.readlines()

    for mol in mol_list:
        mass = 0
        # Split the line by the upper case letters
        mol = mol.strip('\n')
        atom_list = [x for x in re.split('([A-Z][^A-Z]*)', mol) if x]
        # for each component in mol, split the component into element and number
        for comp in atom_list:
            element = re.match('[A-Za-z]*', comp).group()
            number = re.search('[0-9]+', comp)
            if number:
                number = int(number.group())
            else:
                number = 1
            mass += atomic_masses[atomic_numbers[element]] * number
        print(mol, mass)


if __name__ == '__main__':
    main()
