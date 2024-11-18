import os
import tempfile

from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.mixing import MixedCalculator
from sevenn.sevennet_calculator import SevenNetCalculator

from CryoEtchSimulator.relax.mylammps import MyLAMMPS as LAMMPS


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
        damping_type = "damp_bj"
        func_type = "pbe"

        parameters = {
            'pair_style': f'd3 {cutoff_d3} {cutoff_d3_CN} {damping_type} {func_type}',
            'pair_coeff': [f'* * {elements}'],
        }
        if self.lmp_input:
            for k, v in self.lmp_input.items():
                parameters[k] = v

        params_str = 'dict(' + ',\n'.join(f'{k}={v!r}' for k, v in parameters.items()) + ')'

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(params_str)
            params_file = temp_file.name

        calc_settings = {
            'parameters': params_file,
            'keep_alive': True,
            'specorder': specorder,
            # 'keep_tmp_files': True,
            # 'tmp_dir': 'debug',
            # 'always_triclinic': True,
            # 'verbose': True,
        }

        try:
            d3_calculator = LAMMPS(**calc_settings)
        finally:
            os.unlink(params_file)

        return d3_calculator
