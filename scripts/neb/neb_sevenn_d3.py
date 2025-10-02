import time
import os
import sys
from functools import wraps

import yaml
import numpy as np
import matplotlib.pyplot as plt

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
from ase.mep import NEB
from ase.mep import NEBTools


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


def measure_time(func, level=[0]):
    @wraps(func)
    def wrapper(*args, **kwargs):

        line = f"{'| ' * level[0]}Running '{func.__name__}'"
        print(line)
        level[0] += 1

        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        time_taken = end_time - start_time  # Calculate the time taken

        indent_level = '| ' * (level[0] - 1) + '└ '
        level[0] -= 1
        line = f"{indent_level}Finished '{func.__name__}' "
        line += f"({time_taken:.6f} seconds)"
        print(line)

        return result
    return wrapper


@measure_time
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


@measure_time
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
    return calc_gnn


@measure_time
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


@measure_time
def gen_optimizer(atoms, logfile=None, **inputs):
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


@measure_time
def atom_relax(atoms, logfile=None, **inputs):
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

    opt = gen_optimizer(atoms, logfile=logfile, **inputs)
    opt.attach(custom_step_writer)
    run_optimizer(opt, fmax, steps, logfile=logfile)


@measure_time
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


@measure_time
def load_atoms(poscar_path, **inputs):
    '''
    Load atoms from POSCAR file
    '''
    lmp_input = inputs['options'].get('lmp_input')
    atoms = read(poscar_path, format='vasp')
    atoms = atoms[atoms.numbers.argsort()]
    if lmp_input is not None:
        pbc = lmp_input.get('boundary')
        if pbc is not None:
            pbc = [True if i == 'p' else False for i in pbc.split()]
            atoms.set_pbc(pbc)
    return atoms


@measure_time
def relax_end_points(logfile='-', **inputs):
    '''
    Relax end points of NEB calculation
    '''
    path_atoms_initial = inputs['neb']['path_initial']
    path_atoms_final = inputs['neb']['path_final']
    atoms_initial = load_atoms(path_atoms_initial, **inputs)
    atoms_final = load_atoms(path_atoms_final, **inputs)

    calc = gen_calculator(atoms_initial, **inputs)
    atoms_initial.calc = calc
    atom_relax(atoms_initial, logfile=logfile, **inputs)

    calc = gen_calculator(atoms_final, **inputs)
    atoms_final.calc = calc
    atom_relax(atoms_final, logfile=logfile, **inputs)

    return atoms_initial, atoms_final


@measure_time
def make_neb_images(atoms_initial=None, atoms_final=None, **inputs):
    '''
    Generate images for NEB calculation
    '''
    n_images = inputs['neb']['n_images']
    to_continue = inputs['neb']['continue']
    if to_continue:
        path_continue = inputs['neb']['path_continue']
        idx_to_slice = slice(-n_images-2, None)
        neb_images = read(path_continue, index=idx_to_slice)
    else:
        neb_images = [atoms_initial]
        neb_images += [atoms_initial.copy() for i in range(n_images)]
        neb_images += [atoms_final]

    if inputs['neb']['allow_shared_calculator']:
        calc = gen_calculator(neb_images[0], **inputs)
        for image in neb_images[1:-1]:
            image.calc = calc
    else:
        for image in neb_images[1:-1]:
            calc = gen_calculator(image, **inputs)
            image.calc = calc

    neb_params = {
        'climb': inputs['neb']['climb'],
        'allow_shared_calculator': inputs['neb']['allow_shared_calculator'],
    }
    neb = NEB(neb_images, **neb_params)
    if not to_continue:
        neb.interpolate(mic=True)

    return neb


@measure_time
def run_NEB(logfile='-', **inputs):
    '''
    Run NEB calculation
    1. Relax end points
    2. Generate images
    3. Run NEB
    '''
    if inputs['neb']['continue']:
        image_i, image_f = None, None
    else:
        image_i, image_f = relax_end_points(logfile=logfile, **inputs)

    neb = make_neb_images(atoms_initial=image_i, atoms_final=image_f, **inputs)

    opt_type = set_optimizer(**inputs)
    opt_inputs = {
        'logfile': logfile,
        'trajectory': inputs['neb']['trajectory'],
    }
    opt = opt_type(neb, **opt_inputs)

    fmax = inputs['options']['fmax']
    steps = inputs['options']['max_steps']
    run_optimizer(opt, fmax, steps, logfile=logfile)

    summarize_neb(neb.images, **inputs)


def summarize_neb(images, **inputs):
    fig, ax = plt.subplots()
    E = np.array([image.get_potential_energy() for image in images])
    E -= E[0]

    Ef = np.max(E)
    dE = E[-1]
    Er = np.max(E) - dE

    prop_dict = {
        'linestyle': '--',
        'marker': 'o',
        'color': 'black',
    }

    ax.plot(E, **prop_dict)
    title = "$E^{\dagger}_{f}$: " + f"{Ef:.2f} eV, "
    title += "$E^{\dagger}_{r}$: " + f"{Er:.2f} eV, "
    title += "$\Delta E$: " + f"{dE:.2f} eV"
    ax.set_title(title)
    ax.set_xlabel('Image')
    ax.set_ylabel('Relative Energy (eV)')
    fig.tight_layout()
    fig.savefig('neb_band.png')

    # nebtools = NEBTools(images)
    # Ef, dE = nebtools.get_barrier()
    # fig = nebtools.plot_band()
    # fig.savefig('neb_band.png')

    # fig = plt.figure(figsize=(5.5, 4.0))
    # ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
    # nebtools.plot_band(ax)
    # fig.savefig('neb_band_custom.png')

    n_images = inputs['neb']['n_images'] + 2
    path_output = inputs['neb']['path_result']
    write(path_output, images[-n_images:], format='extxyz')


@measure_time
def run_optimizer(opt, fmax, steps, logfile='-'):
    time_init = time.time()
    logfile.write('######################\n')
    logfile.write('##   Relax starts   ##\n')
    logfile.write('######################\n')

    opt.run(fmax=fmax, steps=steps)

    logfile.write(f'\nElapsed time: {time.time()-time_init} s\n\n')
    logfile.write('########################\n')
    logfile.write('##  Relax terminated  ##\n')
    logfile.write('########################\n')


@measure_time
def gen_calculator(atoms, **inputs):
    calc_gnn = gen_gnn_calculator(**inputs)
    if inputs.get('include_d3'):
        calc_d3 = gen_d3_calculator(atoms, **inputs)
        ratio_gnn = 1
        ratio_d3 = 1
        calc = MixedCalculator(calc_gnn, calc_d3, ratio_gnn, ratio_d3)
    else:
        calc = calc_gnn

    return calc


@measure_time
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_yaml>")
        sys.exit()

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    path_log = inputs['path']['log']
    with open(path_log, 'w', buffering=1) as logfile:
        run_NEB(logfile=logfile, **inputs)


if __name__ == "__main__":
    main()
