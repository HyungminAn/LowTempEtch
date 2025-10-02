from pathlib import Path
import time
import os
import sys

from ase.io import read, write
if sys.version_info >= (3, 10):
    from ase.constraints import FixSymmetry
else:
    from ase.spacegroup.symmetrize import FixSymmetry

from CryoEtchSimulator.relax.gen_calc import CalculatorGenerator
from CryoEtchSimulator.relax.gen_opt import OptimizerGenerator
from CryoEtchSimulator.relax.log import RelaxLogger


class ParameterUnsetError(Exception):
    def __init__(self, param_name):
        self.param_name = param_name

    def __str__(self):
        return f"Parameter {self.param_name} is not set"


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


def _get_element_list(atoms):
    specorder = []
    [specorder.append(i) for i in atoms.get_chemical_symbols() if i not in specorder]
    return specorder
