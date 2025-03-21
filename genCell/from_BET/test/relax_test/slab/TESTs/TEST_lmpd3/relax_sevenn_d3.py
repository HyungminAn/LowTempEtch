import time
import os
import sys
import warnings

import yaml
import numpy as np

from ase.io import read, write
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
from ase.optimize import BFGSLineSearch
from ase.optimize import LBFGS
from ase.optimize import LBFGSLineSearch
from ase.optimize import GPMin
from ase.optimize import MDMin
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.units import bar
from ase.spacegroup.symmetrize import FixSymmetry

# warnings.filterwarnings(action='ignore')


def generate_calculator(path_lmp_bin, path_pot, atoms, lmp_input=None):
    os.environ['ASE_LAMMPSRUN_COMMAND'] = path_lmp_bin

    specorder = []
    [specorder.append(i) for i in atoms.get_chemical_symbols()
        if i not in specorder]
    elements = ' '.join(specorder)
    print(elements)

    path_r0ab = "/data2/andynn/lammps_d3/13_rewrite_csv/r0ab_new.csv"
    path_c6ab = "/data2/andynn/lammps_d3/13_rewrite_csv/d3_pars.csv"
    cutoff_d3 = 9000
    cutoff_d3_CN = 1600
    func_type = "pbe"
    damping_type = "d3_damp_bj"

    parameters = {
        'pair_style': f'hybrid/overlay e3gnn d3 {cutoff_d3} {cutoff_d3_CN} \
            {damping_type}',
        'pair_coeff': [
            f'* * e3gnn {path_pot} {elements}',
            f'* * d3 {path_r0ab} {path_c6ab} {func_type} {elements}'],
        'compute': 'my_d3_compute all pressure NULL virial pair/hybrid d3'
    }

    if lmp_input:
        for k, v in lmp_input.items():
            parameters[k] = v

    calculator = LAMMPS(
        parameters=parameters, files=[path_pot], keep_alive=True,
        specorder=specorder,
        )

    return calculator


def save_info(atoms, extxyz_filename, step):
    """
    Save atomic positions to a trajectory file
        and the stress tensor to another file.
    Voigt order: xx, yy, zz, yz, xz, xy
    Saved order: xx, yy, zz, xy, yz, zx
    """

    PotEng = atoms.get_potential_energy()
    volume = atoms.get_volume()
    stress = atoms.get_stress() / (-bar)
    press = np.average(stress[0:3])

    line = f"{step:8d}  {PotEng:15.4f}  {volume:15.4f}  "
    line += f"{press:15.4f}  "
    line += f"{stress[0]:15.4f}  {stress[1]:15.4f}  {stress[2]:15.4f}  "
    line += f"{stress[5]:15.4f}  {stress[3]:15.4f}  {stress[4]:15.4f}\n"

    with open('thermo.dat', 'a') as f:
        f.write(line)

    write(extxyz_filename, atoms, format='extxyz', append=True)


def atom_relax(
        atoms, calc, cell_relax=True, opt_type=None, filter_type=None,
        fmax=None, steps=None, fix_symmetry=False,
        logfile='-', extxyz_file='opt.extxyz', stress_file='stress'):

    if fmax is None:
        print("fmax must be set; exit.")
        sys.exit()

    if steps is None:
        print("max_steps must be set; exit.")
        sys.exit()

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    with open(extxyz_file, 'w') as _:
        pass

    atoms.calc = calc

    if cell_relax:
        ecf = filter_type(atoms)
        opt = opt_type(ecf, logfile=logfile)
    else:
        opt = opt_type(atoms, logfile=logfile)

    # Define a function to be called at each optimization step
    with open('thermo.dat', 'w') as f:
        line = f"{'step':>8s}  {'PotEng':>15s}  {'volume':>15s}  "
        line += f"{'press':>15s}  "
        line += f"{'p_xx':>15s}  {'p_yy':>15s}  {'p_zz':>15s}  "
        line += f"{'p_xy':>15s}  {'p_yz':>15s}  {'p_zx':>15s}\n"
        f.write(line)

    def custom_step_writer():
        step = opt.nsteps
        save_info(atoms, extxyz_file, step)

    opt.attach(custom_step_writer)

    time_init = time.time()
    logfile.write('######################\n')
    logfile.write('##   Relax starts   ##\n')
    logfile.write('######################\n')

    opt.run(fmax=fmax, steps=steps)

    logfile.write(f'\nElapsed time: {time.time()-time_init} s\n\n')
    logfile.write('########################\n')
    logfile.write('##  Relax terminated  ##\n')
    logfile.write('########################\n')
    logfile.close()


def log_initial_structure(atoms, logfile='-'):

    _cell = atoms.get_cell()
    logfile.write("Cell\n")
    for ilat in range(0, 3):
        line = f"{_cell[ilat][0]:.6f}  "
        line += f"{_cell[ilat][1]:.6f}  "
        line += f"{_cell[ilat][2]:.6f}\n"
        logfile.write(line)

    _pbc = atoms.get_pbc()
    logfile.write("PBCs\n")
    logfile.write(f"{_pbc[0]}  {_pbc[1]}  {_pbc[2]}\n")
    _pos = atoms.get_positions()
    _mass = atoms.get_masses()
    _chem_sym = atoms.get_chemical_symbols()
    logfile.write("Atoms\n")
    for i in range(len(_chem_sym)):
        line = f"{_chem_sym[i]}  "
        line += f"{_mass[i]:.6f}  "
        line += f"{_pos[i][0]:.6f}  "
        line += f"{_pos[i][1]:.6f}  "
        line += f"{_pos[i][2]:.6f}\n"
        logfile.write(line)


def main():
    poscar_path, path_yaml = sys.argv[1:3]
    with open(path_yaml, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    path_pot = inputs['path_pot']
    opt_type = inputs['opt_type']
    if opt_type == 'BFGS':
        opt_type = BFGS
    elif opt_type == 'BFGSLineSearch':
        opt_type = BFGSLineSearch
    elif opt_type == 'LBFGS':
        opt_type = LBFGS
    elif opt_type == 'LBFGSLineSearch':
        opt_type = LBFGSLineSearch
    elif opt_type == 'GPMin':
        opt_type = GPMin
    elif opt_type == 'MDMin':
        opt_type = MDMin
    elif opt_type == 'FIRE':
        opt_type = FIRE
    else:
        print('Invalid optimizer type')
        sys.exit()

    cell_relax = inputs['cell_relax']
    filter_type = inputs['filter_type']
    fix_symmetry = inputs['fix_symmetry']
    if cell_relax:
        if filter_type == 'FrechetCellFilter':
            filter_type = FrechetCellFilter
        else:
            print('Invalid filter type')
            sys.exit()

    path_extxyz = inputs['path_extxyz']
    path_relaxed = inputs['path_relaxed']
    path_lmp_bin = inputs['path_lmp_bin']
    max_steps = inputs['max_steps']
    fmax = inputs['fmax']
    lmp_input = inputs.get('lmp_input')

    # atoms = read(poscar_path, format='lammps-data')
    atoms = read(poscar_path, format='vasp')
    atoms = atoms[atoms.numbers.argsort()]
    if lmp_input is not None:
        pbc = lmp_input.get('boundary')
        if pbc is not None:
            pbc = [True if i == 'p' else False for i in pbc.split()]
            atoms.set_pbc(pbc)

    logfile = open('log', 'w', buffering=1)
    log_initial_structure(atoms, logfile)

    calc = generate_calculator(
        path_lmp_bin, path_pot, atoms, lmp_input=lmp_input)

    atom_relax(
        atoms, calc, cell_relax=cell_relax,
        opt_type=opt_type, filter_type=filter_type,
        logfile=logfile, extxyz_file=path_extxyz,
        steps=max_steps, fmax=fmax, fix_symmetry=fix_symmetry
        )

    write(path_relaxed, atoms)


if __name__ == "__main__":
    main()
