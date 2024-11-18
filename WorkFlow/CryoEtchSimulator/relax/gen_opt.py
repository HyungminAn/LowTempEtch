from ase.optimize import BFGS
from ase.optimize import BFGSLineSearch
from ase.optimize import LBFGS
from ase.optimize import LBFGSLineSearch
from ase.optimize import GPMin
from ase.optimize import MDMin
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter


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


class UnavailableParameterError(Exception):
    def __init__(self, param_name, value):
        self.param_name = param_name
        self.value = value

    def __str__(self):
        return f"Parameter {self.param_name} is not available: {self.value}"
