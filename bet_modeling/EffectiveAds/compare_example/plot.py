import matplotlib.pyplot as plt
import numpy as np

class DataLoader:
    def run(self, path_dat):
        data = -np.loadtxt(path_dat, usecols=(2))
        return data

class EffectiveAdsorptionEnergyCalculator:
    def run(self, E_ads, temp_range):
        kB = 8.617333262145e-5  # eV/K
        E_eff = []

        for T in temp_range:
            value = kB*T*np.log(np.sum(np.exp(E_ads/(kB*T)))/len(E_ads))
            E_eff.append(value)

        return E_eff


class DataPlotter:
    def run(self, data):
        plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})
        fig, (ax_hist, ax_line) = plt.subplots(2, 1, figsize=(3.5, 5))

        opts_hist = {
            'bins': 30,
            'alpha': 0.5,
            'range': (0, 0.8),
            'rwidth': 0.9,
            }

        T_min, T_max = 100, 1000
        temp_range = np.arange(T_min, T_max + 1, 1)
        eaec = EffectiveAdsorptionEnergyCalculator()

        for key, data in data.items():
            ax_hist.hist(data, label=key, **opts_hist)
            ax_line.plot(temp_range,
                         eaec.run(data, temp_range),
                         label=key)

        ax_hist.legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.25),
                       ncol=2,
                       frameon=False)
        ax_hist.set_xlim(0, 0.8)
        ax_hist.set_xlabel('Adsorption energy (eV)')
        ax_hist.set_ylabel('Count')
        ax_hist.text(-0.2, 1.1, '(a)',
                     transform=ax_hist.transAxes, fontsize=10,
                     va='top', ha='left')

        ax_line.legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.25),
                       ncol=2,
                       frameon=False)
        ax_line.set_xlim(T_min, T_max)
        ax_line.set_xlabel('Temperature (K)')
        ax_line.set_ylabel(r'$E_\text{eff}^\text{ads}$')
        ax_line.text(-0.2, 1.1, '(b)',
                     transform=ax_line.transAxes, fontsize=10,
                     va='top', ha='left')

        fig.tight_layout()
        fig.savefig('result.png', dpi=200)
        fig.savefig('result.pdf')
        fig.savefig('result.eps')

def main():
    dl = DataLoader()
    path_dict = {
            'SiO$_2$': 'energy_HF_on_SiO2.dat',
            'SiO$_2$ + IF$_5$ 1ML': 'energy_HF_on_IF5_on_SiO2.dat',
            }
    data = {}
    for key, path in path_dict.items():
        data[key] = dl.run(path)

    dp = DataPlotter()
    dp.run(data)

if __name__ == "__main__":
    main()
