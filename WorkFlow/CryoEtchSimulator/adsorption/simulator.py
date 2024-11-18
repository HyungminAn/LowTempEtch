import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from CryoEtchSimulator.genCell.simulator import CellGenerator
from CryoEtchSimulator.relax.simulator import RelaxSimulator
from CryoEtchSimulator.md.simulator import MolecularDynamicsSimulator


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

        self.fix_h = inputs['constraint']['fix_bottom_height']

        self.cellgen_tolerance = inputs['cell_generation']['tolerance']
        self.cellgen_path_packmol = inputs['cell_generation']['path']['packmol']
        self.cellgen_path_input = inputs['cell_generation']['path']['input']
        self.cellgen_path_output = inputs['cell_generation']['path']['output']
        self.cellgen_path_log = inputs['cell_generation']['path']['log']

        self.path_lmp = inputs['relax']['path']['lmp_bin']
        self.path_md_potential = inputs['relax']['path']['pot']['7net_pt'].get(inputs['relax']['model'])

        self.energy = {
            'mol': _read_energy_from_thermo_dat(
                Path(self.path_etchant).parent/'thermo.dat'),
        }

    def run(self):
        self._process_molecules('additive')
        self._process_molecules('etchant')
        self._summarize_results()

    def _process_molecules(self, molecule_type):
        is_additive = molecule_type == 'additive'
        n_repeat = self.additive_n_repeat if is_additive else self.etchant_n_repeat
        folder_name = '01_additive' if is_additive else '02_etchant'
        poscar_path = self.path_slab if is_additive else self.slab

        input_dict_cellgen = self._create_cellgen_input_dict(
                self.additive_n_layer, self.etchant_n_layer, poscar_path, is_additive)

        for i in range(n_repeat):
            dst = self.dst / f'{folder_name}/{i}'
            input_dict_cellgen['path']['path_dst'] = dst
            self._generate_and_process_cell(dst, input_dict_cellgen, is_additive)

        self._update_slab(folder_name, n_repeat, is_additive)

    def _create_cellgen_input_dict(
            self, n_layer_addi, n_layer_etchant, poscar_path, is_additive):
        return {
            'params': {
                'tolerance': self.cellgen_tolerance,
                'layer_ADDI': n_layer_addi,
                'layer_HF': n_layer_etchant,
                'n_mol_HF': None if is_additive else 1,
                'n_mol_additive': None if is_additive else 0,
            },
            'path': {
                'path_mol': self.path_additive,
                'path_poscar': poscar_path,
                'path_HF': self.path_etchant,
                'path_packmol': self.cellgen_path_packmol,
                'path_input': self.cellgen_path_input,
                'path_output': self.cellgen_path_output,
                'path_log': self.cellgen_path_log,
                'path_dst': None,
            },
        }

    def _generate_and_process_cell(self, dst, input_dict_cellgen, is_additive):
        if not os.path.exists(dst/'POSCAR'):
            self.cellgen = CellGenerator(input_dict_cellgen)
            self.cellgen.generate()

        if not is_additive:
            self.md_flag = False

        if self.md_flag and is_additive:
            self._run_molecular_dynamics(dst)

        if not os.path.exists(dst/'POSCAR_relaxed'):
            self._run_relaxation(dst)

    def _run_molecular_dynamics(self, dst):
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

    def _run_relaxation(self, dst):
        self.relaxSimulator = RelaxSimulator(self.inputs['relax'])
        if self.md_flag:
            self.relaxSimulator.run(dst/'POSCAR_after_md', dst)
        else:
            self.relaxSimulator.run(dst/'POSCAR', dst)

    def _update_slab(self, folder_name, n_repeat, is_additive):
        energies = [
            (i, _read_energy_from_thermo_dat(self.dst/f'{folder_name}/{i}/thermo.dat'))
            for i in range(n_repeat)
        ]
        if is_additive:
            idx_slab, self.energy['slab'] = min(energies, key=lambda x: x[1])
            self.slab = os.path.abspath(
                self.dst/f'{folder_name}/{idx_slab}/POSCAR_relaxed')
        else:
            self.energy['slabMol'] = [energy for _, energy in energies]

    def _summarize_results(self):
        E_before = self.energy['slab'] + self.energy['mol']
        E_ads = [E - E_before for E in self.energy['slabMol']]
        temp_range = np.linspace(200, 400)

        plotter = EffectiveAdsorptionEnergyPlotter(self.dst, E_ads, temp_range)
        plotter.plot()

        self._save_data()

    def _save_data(self):
        with open('result_adsorption.yaml', 'w') as f:
            yaml.dump(self.energy, f)


class EffectiveAdsorptionEnergyPlotter():
    def __init__(self, dst, energies, temp_range):
        self.dst = dst
        self.energies = energies
        self.temp_range = temp_range

    def plot(self):
        self._get_effective_adsorption_energy()

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        ax.plot(self.temp_range, self.E_eff)

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Effective adsorption energy (eV)')

        fig.tight_layout()
        fig.savefig('E_ads_eff.png')

        self._save_data()

    def _get_effective_adsorption_energy(self):
        kB = 8.617333262145e-5  # eV/K
        E_eff = []

        for T in self.temp_range:
            value = kB*T*np.log(np.sum(np.exp(self.energies/(kB*T)))/len(self.energies))
            E_eff.append(value)

        self.E_eff = E_eff

    def _save_data(self):
        with open('dat', 'w') as f:
            for T, E in zip(self.temp_range, self.E_eff):
                f.write(f'{T} {E}\n')

def _read_energy_from_thermo_dat(path_thermo_dat):
    with open(path_thermo_dat, 'r') as f:
        lines = f.readlines()
        energy = float(lines[-1].split()[1])
    return energy
