from pathlib import Path
import time
import os
import sys
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
if sys.version_info >= (3, 10):
    from ase.constraints import FixSymmetry
else:
    from ase.spacegroup.symmetrize import FixSymmetry
from ase.calculators.mixing import MixedCalculator
from sevenn.sevennet_calculator import SevenNetCalculator


class UnavailableParameterError(Exception):
    def __init__(self, param_name, value):
        self.param_name = param_name
        self.value = value

    def __str__(self):
        return f"Parameter {self.param_name} is not available: {self.value}"


class ParameterUnsetError(Exception):
    def __init__(self, param_name):
        self.param_name = param_name

    def __str__(self):
        return f"Parameter {self.param_name} is not set"


def gen_d3_calculator(atoms, **inputs):
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

    calc_settings = {
        'parameters': parameters,
        'keep_alive': True,
        'specorder': specorder,
        # 'keep_tmp_files': True,
        # 'tmp_dir': 'debug',
        # 'always_triclinic': True,
        # 'verbose': True,
    }

    d3_calculator = LAMMPS(**calc_settings)
    return d3_calculator


def gen_gnn_calculator(**inputs):
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


def save_info(atoms, thermo_dat, extxyz_filename, step):
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

    with open(thermo_dat, 'a') as f:
        f.write(line)

    write(extxyz_filename, atoms, format='extxyz', append=True)


def gen_optimizer(atoms, logfile=None, **inputs):
    '''
    Generate an optimizer for the given *atoms*
    '''
    opt_type = set_optimizer(**inputs)
    cell_relax = inputs['options']['cell_relax']
    filter_type = inputs['options']['filter_type']
    if cell_relax:
        if filter_type == 'FrechetCellFilter':
            filter_type = FrechetCellFilter
        else:
            raise UnavailableParameterError('filter_type', filter_type)

        filter_options = inputs['options'].get('filter_options')
        if filter_options is not None:
            ecf = filter_type(atoms, **filter_options)
        else:
            ecf = filter_type(atoms)
        opt = opt_type(ecf, logfile=logfile)
    else:
        opt = opt_type(atoms, logfile=logfile)

    return opt


def run_optimizer(opt, fmax, steps, logfile='-'):
    '''
    Run the optimizer *opt* with given *fmax* and *steps*
    '''
    time_init = time.time()
    logfile.write('######################\n')
    logfile.write('##   Relax starts   ##\n')
    logfile.write('######################\n')

    opt.run(fmax=fmax, steps=steps)

    logfile.write(f'\nElapsed time: {time.time()-time_init} s\n\n')
    logfile.write('########################\n')
    logfile.write('##  Relax terminated  ##\n')
    logfile.write('########################\n')


def atom_relax(atoms, dst=None, logfile=None, **inputs):
    '''
    Run structure optimization for given *atoms*
    '''
    if atoms.calc is None:
        raise ParameterUnsetError('calculator')

    fmax = inputs['options']['fmax']
    if fmax is None:
        raise ParameterUnsetError('fmax')

    steps = inputs['options']['max_steps']
    if steps is None:
        raise ParameterUnsetError('max_steps')

    fix_symmetry = inputs['options']['fix_symmetry']
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    traj = dst/inputs['path']['traj']
    thermo = dst/'thermo.dat'
    with open(traj, 'w') as _:
        pass

    # Define a function to be called at each optimization step
    with open(thermo, 'w') as f:
        line = f"{'step':>8s}  {'PotEng':>15s}  {'volume':>15s}  "
        line += f"{'press':>15s}  "
        line += f"{'p_xx':>15s}  {'p_yy':>15s}  {'p_zz':>15s}  "
        line += f"{'p_xy':>15s}  {'p_yz':>15s}  {'p_zx':>15s}\n"
        f.write(line)

    def custom_step_writer():
        step = opt.nsteps
        save_info(atoms, thermo, traj, step)

    opt = gen_optimizer(atoms, logfile=logfile, **inputs)
    opt.attach(custom_step_writer)
    run_optimizer(opt, fmax, steps, logfile=logfile)


def log_initial_structure(atoms, logfile='-'):
    '''
    Log the initial structure to the *logfile*
    '''
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
    opt_dict = {
        'BFGS': BFGS,
        'BFGSLineSearch': BFGSLineSearch,
        'LBFGS': LBFGS,
        'LBFGSLineSearch': LBFGSLineSearch,
        'GPMin': GPMin,
        'MDMin': MDMin,
        'FIRE': FIRE,
    }
    opt_type = opt_dict.get(opt_type)
    if opt_type is not None:
        return opt_type
    else:
        raise UnavailableParameterError('opt_type', opt_type)


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


def gen_calculator(atoms, **inputs):
    '''
    Generate a calculator for the given *atoms*
    '''
    calc_gnn = gen_gnn_calculator(**inputs)
    if inputs.get('include_d3'):
        calc_d3 = gen_d3_calculator(atoms, **inputs)
        ratio_gnn = 1
        ratio_d3 = 1
        calc = MixedCalculator(calc_gnn, calc_d3, ratio_gnn, ratio_d3)
    else:
        calc = calc_gnn

    return calc


def relax(poscar_path, path_dst, **inputs):
    '''
    Relax the structure given by *poscar_path*
    '''
    inputs = inputs['relax']
    dst = Path(path_dst)
    with open(dst/'log', 'w') as logfile:
        atoms = load_atoms(poscar_path, **inputs)
        log_initial_structure(atoms, logfile)

        calc = gen_calculator(atoms, **inputs)
        atoms.calc = calc

        atom_relax(atoms, dst=dst, logfile=logfile, **inputs)

        path_output = dst/inputs['path']['output']
        write(path_output, atoms)
