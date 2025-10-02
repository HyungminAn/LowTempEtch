import os

import matplotlib.pyplot as plt
import numpy as np

import yaml


def read_energy(path_slab):
    folder_list = os.listdir(path_slab)
    result = {}
    for folder in folder_list:
        path = os.path.join(path_slab, folder, "thermo.dat")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            lines = f.readlines()
            energy = float(lines[-1].split()[1])
            result[int(folder)] = energy
    return result


def read_energy_mol():
    src = "/data2/andynn/LowTempEtch/03_gases/benchmark/chgTot/HF"
    with open(os.path.join(src, "thermo.dat"), "r") as f:
        lines = f.readlines()
        energy = float(lines[-1].split()[1])
    return energy


def plot_adsorption_energy(src):
    gas_type = os.path.basename(src)
    temp_range = np.arange(200, 300, 1)
    target_temp = 213
    dst = "result_adsorption"
    if os.path.exists(f"{dst}/ads_{gas_type}.yaml"):
        return

    path_slab = os.path.join(src, "run", "01_additive")
    energy_slab = read_energy(path_slab)
    E_slab = min([i for i in energy_slab.values()])

    path_slabmol = os.path.join(src, "run", "02_etchant")
    energy_slabmol = read_energy(path_slabmol)

    energy_mol = read_energy_mol()
    E_ads = [(i, - (E - E_slab - energy_mol)) for i, E in energy_slabmol.items()]
    E_ads_dict = dict(E_ads)

    plotter = EffectiveAdsorptionEnergyPlotter(E_ads, temp_range, gas_type, dst)
    plotter.plot()

    E_ads_eff = plotter.get_target_value(target_temp)

    result = {}
    result['E_slab'] = energy_slab
    result['E_slab_min'] = E_slab
    result['E_slabmol'] = energy_slabmol
    result['E_ads'] = E_ads_dict
    result['E_mol'] = energy_mol
    result['E_ads_eff'] = E_ads_eff
    with open(f"{dst}/ads_{gas_type}.yaml", "w") as f:
        yaml.dump(result, f, sort_keys=True)


class EffectiveAdsorptionEnergyPlotter():
    def __init__(self, energies, temp_range, gas_type, dst):
        self.energies = energies
        self.temp_range = temp_range
        self.gas_type = gas_type
        self.dst = dst
        os.makedirs(self.dst, exist_ok=True)

    def plot(self):
        self._get_effective_adsorption_energy()

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        x = [T for (T, _) in self.E_eff]
        y = [E for (_, E) in self.E_eff]
        ax.plot(x, y)

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Effective adsorption energy (eV)')

        fig.tight_layout()
        fig.savefig(f"{self.dst}/ads_{self.gas_type}.png")

        self._save_data()

    def get_target_value(self, T):
        if self.E_eff is None:
            self._get_effective_adsorption_energy()

        for (T_, E) in self.E_eff:
            if T == T_:
                return float(E)

    def _get_effective_adsorption_energy(self):
        kB = 8.617333262145e-5  # eV/K
        E_i = np.array([E for (_, E) in self.energies])
        E_eff = [(T, kB*T*np.log(np.average(np.exp(E_i/(kB*T))))) for T in self.temp_range]
        self.E_eff = E_eff

    def _save_data(self):
        with open(f'{self.dst}/E_ads_{self.gas_type}.dat', 'w') as f:
            for T, E in zip(self.temp_range, self.E_eff):
                f.write(f'{T} {E}\n')
