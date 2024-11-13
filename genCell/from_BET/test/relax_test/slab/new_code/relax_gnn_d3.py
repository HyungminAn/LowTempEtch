import time
import os
import sys

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
from ase.calculators.mixing import MixedCalculator
from sevenn.sevennet_calculator import SevenNetCalculator


def generate_d3_calculator(atoms, **inputs):
    '''
    D3 calculator for LAMMPS
    '''
    path_lmp_bin = inputs['path']['lmp_bin']
    path_r0ab = inputs['path']['d3']['r0ab']
    path_c6ab = inputs['path']['d3']['c6ab']
    lmp_input = inputs['options'].get('lmp_input')

    os.environ['ASE_LAMMPSRUN_COMMAND'] = path_lmp_bin

    specorder = []
    [specorder.append(i) for i in atoms.get_chemical_symbols()
        if i not in specorder]
    elements = ' '.join(specorder)
    print(elements)

    cutoff_d3 = 9000
    cutoff_d3_CN = 1600
    func_type = "pbe"
    damping_type = "d3_damp_bj"

    parameters = {
        'pair_style': f'd3 {cutoff_d3} {cutoff_d3_CN} {damping_type}',
        'pair_coeff': [f'* * {path_r0ab} {path_c6ab} {func_type} {elements}'],
    }

    if lmp_input:
        for k, v in lmp_input.items():
            parameters[k] = v

    d3_calculator = LAMMPS(
        parameters=parameters, keep_alive=True, specorder=specorder,
        )

    return d3_calculator


def load_calc_gnn(**inputs):
    '''
    GNN calculator for LAMMPS
    '''
    model = inputs['model']
    model_path = inputs['path']['pot']['7net'].get(model)
    if model_path is None:
        raise ValueError(f"Model {model} not found")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    calc_gnn = SevenNetCalculator(model=model_path)
    print(f"GNN model {model} loaded: {model_path}")
    return calc_gnn


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


def atom_relax(atoms, logfile=None, **inputs):
    cell_relax = inputs['options']['cell_relax']
    filter_type = inputs['options']['filter_type']
    fix_symmetry = inputs['options']['fix_symmetry']
    opt_type = set_optimizer(**inputs)
    if cell_relax:
        if filter_type == 'FrechetCellFilter':
            filter_type = FrechetCellFilter
        else:
            print('Invalid filter type')
            sys.exit()

        ecf = filter_type(atoms)
        opt = opt_type(ecf, logfile=logfile)
    else:
        opt = opt_type(atoms, logfile=logfile)

    fmax = inputs['options']['fmax']
    if fmax is None:
        print("fmax must be set; exit.")
        sys.exit()

    steps = inputs['options']['max_steps']
    if steps is None:
        print("max_steps must be set; exit.")
        sys.exit()

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    traj = inputs['path']['traj']
    with open(traj, 'w') as _:
        pass

    # Define a function to be called at each optimization step
    with open('thermo.dat', 'w') as f:
        line = f"{'step':>8s}  {'PotEng':>15s}  {'volume':>15s}  "
        line += f"{'press':>15s}  "
        line += f"{'p_xx':>15s}  {'p_yy':>15s}  {'p_zz':>15s}  "
        line += f"{'p_xy':>15s}  {'p_yz':>15s}  {'p_zx':>15s}\n"
        f.write(line)

    def custom_step_writer():
        step = opt.nsteps
        save_info(atoms, traj, step)

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


def set_optimizer(**inputs):
    opt_type = inputs['options']['opt_type']
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

    return opt_type


def load_atoms(poscar_path, **inputs):
    lmp_input = inputs['options'].get('lmp_input')
    atoms = read(poscar_path, format='vasp')
    atoms = atoms[atoms.numbers.argsort()]
    if lmp_input is not None:
        pbc = lmp_input.get('boundary')
        if pbc is not None:
            pbc = [True if i == 'p' else False for i in pbc.split()]
            atoms.set_pbc(pbc)
    return atoms


def main():
    poscar_path, path_yaml = sys.argv[1:3]
    with open(path_yaml, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    path_log = inputs['path']['log']
    logfile = open(path_log, 'w', buffering=1)

    atoms = load_atoms(poscar_path, **inputs)
    log_initial_structure(atoms, logfile)

    model_type = inputs['model']
    calc_gnn = load_calc_gnn(**inputs)
    calc_d3 = generate_d3_calculator(atoms, **inputs)
    calc = MixedCalculator(calc_gnn, calc_d3, 1, 1)
    atoms.calc = calc

    atom_relax(atoms, logfile=logfile, **inputs)
    path_output = inputs['path']['output']
    write(path_output, atoms)


if __name__ == "__main__":
    main()
