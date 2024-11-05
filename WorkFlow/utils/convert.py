from ase.io import read, write
from ase.constraints import FixAtoms


def pos2lmp(path_poscar, path_dst, elem_order):
    '''
    Read VASP POSCAR and convert to lammps data format
    '''
    poscar = read(path_poscar)
    write(path_dst, poscar, format='lammps-data', specorder=elem_order)


def lmp2pos(path_lmpdat, path_dst, fix_bottom_height=None):
    '''
    Read lammps data file and convert to VASP POSCAR format
    '''
    lmpdat = read(path_lmpdat, format='lammps-data', atom_style='atomic')
    if fix_bottom_height:
        c = FixAtoms(indices=[
            atom.index for atom in lmpdat
            if atom.position[2] <= fix_bottom_height]
        )
        lmpdat.set_constraint(c)
    write(path_dst, lmpdat, format='vasp', sort=True)
