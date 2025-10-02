import os
from pathlib import Path

from ase.io import read, write

from CryoEtchSimulator.md.simulator import MolecularDynamicsSimulator
from CryoEtchSimulator.diffusion.msd import MeanSquaredDisplacementPlotter
from CryoEtchSimulator.diffusion.diff_coeff import DiffusionCoefficientPlotter


class DiffusionSimulator():
    def __init__(self, inputs, slab):
        self.inputs = inputs
        self.slab = slab
        self.dst = Path(inputs['dst'])
        self.n_repeat = inputs['diffusion']['n_repeat']
        self.md_temp = inputs['diffusion']['md_temp']
        self.md_time = inputs['diffusion']['md_time']

        self.fix_h = inputs['constraint']['fix_bottom_height']
        self.path_lmp = inputs['relax']['path']['lmp_bin']
        self.path_md_potential = inputs['relax']['path']['pot']['7net_pt'].get(inputs['relax']['model'])

        self.slab_replicate = (2, 2, 1)

    def run(self):
        self._replicate_slab()
        self._run_md()
        self._summarize_results()

    def _run_md(self):
        input_dict_md = {
            'dst': None,
            'path_poscar': self.slab,
            'path_lmp': self.path_lmp,
            'path_potential': self.path_md_potential,
            'md_time': self.md_time,
            'md_temp': None,
            'fix_h': self.fix_h,
        }
        for temp in self.md_temp:
            input_dict_md['md_temp'] = temp

            for repeat in range(self.n_repeat):
                dst = self.dst/f'diffusion/{temp}/{repeat}'
                os.makedirs(dst, exist_ok=True)
                input_dict_md['dst'] = dst
                MDrunner = MolecularDynamicsSimulator(input_dict_md)
                MDrunner.run()

    def _summarize_results(self):
        path_dict = {temp: [] for temp in self.md_temp}
        for temp in self.md_temp:
            for repeat in range(self.n_repeat):
                dst = self.dst/f'diffusion/{temp}/{repeat}/dump.lammps'
                msd = MeanSquaredDisplacementPlotter('MD', dst)
                msd.run()

                path_dict[temp].append(self.dst/f'diffusion/{temp}/{repeat}/msd_avg.dat')

        diff = DiffusionCoefficientPlotter(self.dst/'diffusion', path_dict)
        diff.run()

    def _replicate_slab(self):
        slab = read(self.slab)
        slab = slab.repeat(self.slab_replicate)
        dst = self.dst/'diffusion/POSCAR_rep'
        os.makedirs(dst.parent, exist_ok=True)
        write(dst, slab, format='vasp', sort=True)
        self.slab = dst
