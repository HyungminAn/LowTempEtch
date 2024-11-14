import os
from pathlib import Path

from genCell.simulator import CellGenerator
from relax.simulator import RelaxSimulator
from md.simulator import MolecularDynamicsSimulator


class AdsorptionSimulator():
    def __init__(self, inputs):
        self.inputs = inputs
        self.dst = Path(inputs['dst'])

        self.path_slab = inputs['adsorption']['paths']['slab']
        self.path_additive = inputs['adsorption']['paths']['additive']
        self.path_etchant = inputs['adsorption']['paths']['etchant']

        self.additive_name = inputs['adsorption']['mol_info']['additive']['name']
        self.additive_n_layer = inputs['adsorption']['mol_info']['additive']['n_layer']
        self.additive_n_repeat = inputs['adsorption']['mol_info']['additive']['n_repeat']

        self.etchant_name = inputs['adsorption']['mol_info']['etchant']['name']
        self.etchant_n_layer = inputs['adsorption']['mol_info']['etchant']['n_layer']
        self.etchant_n_repeat = inputs['adsorption']['mol_info']['etchant']['n_repeat']

        self.md_flag = inputs['adsorption']['md']['flag']
        self.md_time = inputs['adsorption']['md']['time']
        self.md_temp = inputs['adsorption']['md']['temp']

        self.perturb_flag = inputs['adsorption']['perturb']['flag']
        self.perturb_scale = inputs['adsorption']['perturb']['scale']
        self.perturb_cutoff = inputs['adsorption']['perturb']['cutoff']

        self.fix_h = inputs['constraint']['fix_bottom_height']

        self.cellgen_tolerance = inputs['cell_generation']['tolerance']
        self.cellgen_path_packmol = inputs['cell_generation']['path']['packmol']
        self.cellgen_path_input = inputs['cell_generation']['path']['input']
        self.cellgen_path_output = inputs['cell_generation']['path']['output']
        self.cellgen_path_log = inputs['cell_generation']['path']['log']

        self.path_lmp = inputs['relax']['path']['lmp_bin']
        self.path_md_potential = inputs['relax']['path']['pot']['7net_pt'].get(inputs['relax']['model'])

    def run(self):
        self._add_additive_to_slab()
        self._add_etchant_to_slab()
        self._summarize_results()

    def _add_additive_to_slab(self):
        input_dict_cellgen = {
            'params': {
                'tolerance': self.cellgen_tolerance,
                'layer_ADDI': self.additive_n_layer,
                'layer_HF': 0,
                },
            'path': {
                'path_mol': self.path_additive,
                'path_poscar': self.path_slab,
                'path_HF': self.path_etchant,
                'path_packmol': self.cellgen_path_packmol,
                'path_input': self.cellgen_path_input,
                'path_output': self.cellgen_path_output,
                'path_log': self.cellgen_path_log,
                'path_dst': None,
                },
            }

        for i in range(self.additive_n_repeat):
            dst = self.dst/f'01_additive/{i}'
            input_dict_cellgen['path']['path_dst'] = dst
            if not os.path.exists(dst):
                self.cellgen = CellGenerator(input_dict_cellgen)
                self.cellgen.generate()

            if self.md_flag:
                input_dict_md = {
                    'dst': dst,
                    'path_poscar': f'{dst}/POSCAR',
                    'path_lmp': self.path_lmp,
                    'path_potential': self.path_md_potential,
                    'md_time': self.md_time,
                    'md_temp': self.md_temp,
                    'fix_h': self.fix_h,
                }
                self.md = MolecularDynamicsSimulator(input_dict_md)
                self.md.run()

            if not os.path.exists(dst/'POSCAR_relaxed'):
                self.relaxSimulator = RelaxSimulator(self.inputs['relax'])
                self.relaxSimulator.run(dst/'POSCAR_after_md', dst)

    def _select_slab_with_minimum_energy(self):
        pass

    def _add_etchant_to_slab(self):
        pass

    def _summarize_results(self):
        pass
