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


class CalculatorGenerator():
    def __init__(self, inputs):
        self.lmp_bin = inputs['path']['lmp_bin']
        self.lmp_input = inputs['options'].get('lmp_input')
        self.model = inputs['model']
        self.model_path = inputs['path']['pot']['7net'].get(self.model)
        self.d3_flag = inputs['include_d3']

    def generate(self, elem_list):
        '''
        Generate a calculator for the given *elem_list*
        '''
        calc_gnn = self._gen_gnn_calculator()
        if self.d3_flag:
            calc_d3 = self._gen_d3_calculator(elem_list)
            ratio_gnn = 1
            ratio_d3 = 1
            calc = MixedCalculator(calc_gnn, calc_d3, ratio_gnn, ratio_d3)
        else:
            calc = calc_gnn

        return calc

    def _gen_gnn_calculator(self):
        '''
        GNN calculator for LAMMPS
        '''
        if self.model_path is None:
            raise ValueError(f"Model {self.model} not found")

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")

        calc_gnn = SevenNetCalculator(model=self.model_path)
        print(f"GNN model {self.model} loaded: {self.model_path}")
        return calc_gnn

    def _gen_d3_calculator(self, specorder):
        '''
        D3 calculator for LAMMPS
        '''
        os.environ['ASE_LAMMPSRUN_COMMAND'] = self.lmp_bin

        elements = ' '.join(specorder)
        print(elements)

        cutoff_d3 = 9000
        cutoff_d3_CN = 1600
        func_type = "pbe"
        damping_type = "d3_damp_bj"

        parameters = {
            'pair_style': f'd3 {cutoff_d3} {cutoff_d3_CN} {damping_type} {func_type}',
            'pair_coeff': [f'* * {elements}'],
        }
        if self.lmp_input:
            for k, v in self.lmp_input.items():
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


class OptimizerGenerator():
    def __init__(self, inputs):
        self.opt_type = inputs['options']['opt_type']
        self.cell_relax = inputs['options']['cell_relax']
        self.filter_type = inputs['options']['filter_type']
        self.filter_options = inputs['options'].get('filter_options')

    def generate(self, atoms, logger):
        '''
        Generate an optimizer for the given *atoms*
        '''
        opt_type = self._set_optimizer()
        if self.cell_relax:
            if self.filter_type == 'FrechetCellFilter':
                filter_type = FrechetCellFilter
            else:
                raise UnavailableParameterError('filter_type', self.filter_type)

            # filter_options = inputs['options'].get('filter_options')
            if self.filter_options is not None:
                ecf = filter_type(atoms, **self.filter_options)
            else:
                ecf = filter_type(atoms)
            opt = opt_type(ecf, logfile=logger.logfile)
        else:
            opt = opt_type(atoms, logfile=logger.logfile)

        def _custom_step_writer():
            step = opt.nsteps
            logger._save_info(atoms, step)

        opt.attach(_custom_step_writer)
        return opt

    def _set_optimizer(self):
        opt_dict = {
            'BFGS': BFGS,
            'BFGSLineSearch': BFGSLineSearch,
            'LBFGS': LBFGS,
            'LBFGSLineSearch': LBFGSLineSearch,
            'GPMin': GPMin,
            'MDMin': MDMin,
            'FIRE': FIRE,
        }
        opt_type = opt_dict.get(self.opt_type)
        if opt_type is not None:
            return opt_type
        else:
            raise UnavailableParameterError('opt_type', opt_type)


class RelaxSimulator():
    def __init__(self, inputs):
        self.inputs = inputs
        self.lmp_input = inputs['options'].get('lmp_input')
        self.fmax = inputs['options']['fmax']
        self.steps = inputs['options']['max_steps']
        self.fix_symmetry = inputs['options']['fix_symmetry']
        self.path_traj = inputs['path']['traj']
        self.path_thermo = 'thermo.dat'
        self.path_output = inputs['path']['output']

    def run(self, poscar_path, path_dst):
        '''
        Relax the structure given by *poscar_path*
        '''
        self.logger = RelaxLogger()
        self.dst = Path(path_dst)
        os.makedirs(self.dst, exist_ok=True)
        with open(self.dst/'log', 'w') as logfile:
            self.logger._set_logfile(logfile)
            self._load_atoms(poscar_path)
            self.logger._log_initial_structure(self.atoms)

            calc = CalculatorGenerator(self.inputs).generate(_get_element_list(self.atoms))
            self.atoms.calc = calc

            self._atom_relax()

            write(self.dst/self.path_output, self.atoms, format='vasp')
        self.logfile = None

    def _load_atoms(self, poscar_path):
        atoms = read(poscar_path, format='vasp')
        atoms = atoms[atoms.numbers.argsort()]
        if self.lmp_input is not None:
            pbc = self.lmp_input.get('boundary')
            if pbc is not None:
                pbc = [True if i == 'p' else False for i in pbc.split()]
                atoms.set_pbc(pbc)
        self.atoms = atoms

    def _atom_relax(self):
        '''
        Run structure optimization for given *atoms*
        '''
        if self.atoms.calc is None:
            raise ParameterUnsetError('calculator')
        if self.fmax is None:
            raise ParameterUnsetError('fmax')
        if self.steps is None:
            raise ParameterUnsetError('max_steps')
        if self.fix_symmetry:
            self.atoms.set_constraint(FixSymmetry(self.atoms))

        traj = self.dst/self.path_traj
        thermo = self.dst/self.path_thermo
        self.logger._initialize_relax_log(traj, thermo)

        opt = OptimizerGenerator(self.inputs).generate(self.atoms, self.logger)
        self.opt = opt
        self._run_optimizer()

    def _run_optimizer(self):
        '''
        Run the optimizer *opt* with given *fmax* and *steps*
        '''
        logfile = self.logger.logfile
        w = logfile.write

        time_init = time.time()
        w('######################\n')
        w('##   Relax starts   ##\n')
        w('######################\n')

        self.opt.run(fmax=self.fmax, steps=self.steps)

        w(f'\nElapsed time: {time.time()-time_init} s\n\n')
        w('########################\n')
        w('##  Relax terminated  ##\n')
        w('########################\n')


class RelaxLogger():
    def __init__(self):
        self.logfile = None

    def _set_logfile(self, logfile):
        self.logfile = logfile

    def _log_initial_structure(self, atoms):
        '''
        Log the initial structure to the *logfile*
        '''
        if self.logfile is None:
            raise ParameterUnsetError('logfile')

        w = self.logfile.write
        _cell = atoms.get_cell()
        w("Cell\n")
        for ilat in range(0, 3):
            line = f"{_cell[ilat][0]:.6f}  "
            line += f"{_cell[ilat][1]:.6f}  "
            line += f"{_cell[ilat][2]:.6f}\n"
            w(line)

        _pbc = atoms.get_pbc()
        w("PBCs\n")
        w(f"{_pbc[0]}  {_pbc[1]}  {_pbc[2]}\n")
        _pos = atoms.get_positions()
        _mass = atoms.get_masses()
        _chem_sym = atoms.get_chemical_symbols()
        w("Atoms\n")
        for i in range(len(_chem_sym)):
            line = f"{_chem_sym[i]}  "
            line += f"{_mass[i]:.6f}  "
            line += f"{_pos[i][0]:.6f}  "
            line += f"{_pos[i][1]:.6f}  "
            line += f"{_pos[i][2]:.6f}\n"
            w(line)

    def _initialize_relax_log(self, traj, thermo):
        self.traj = traj
        self.thermo = thermo

        with open(traj, 'w') as _:
            pass

        # Define a function to be called at each optimization step
        with open(thermo, 'w') as f:
            line = f"{'step':>8s}  {'PotEng':>15s}  {'volume':>15s}  "
            line += f"{'press':>15s}  "
            line += f"{'p_xx':>15s}  {'p_yy':>15s}  {'p_zz':>15s}  "
            line += f"{'p_xy':>15s}  {'p_yz':>15s}  {'p_zx':>15s}\n"
            f.write(line)

    def _save_info(self, atoms, step):
        """
        Save atomic positions to a trajectory file
            and the stress tensor to another file.
        Voigt order: xx, yy, zz, yz, xz, xy
        Saved order: xx, yy, zz, xy, yz, zx
        """
        if self.traj is None:
            raise ParameterUnsetError('traj')
        if self.thermo is None:
            raise ParameterUnsetError('thermo')
        thermo_dat = self.thermo
        extxyz_filename = self.traj

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


def _get_element_list(atoms):
    specorder = []
    [specorder.append(i) for i in atoms.get_chemical_symbols() if i not in specorder]
    return specorder
