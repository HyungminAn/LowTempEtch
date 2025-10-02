import os
import math
from dataclasses import dataclass
import yaml

import numpy as np
import matplotlib.pyplot as plt

from msd import MeanSquaredDisplacementPlotter


@dataclass
class Constants:
    KB = 8.617E-05  # eV/K
    DIM = 2
    A2_TO_CM2 = 1E-16
    PS_TO_S = 1E-12


class DiffCoeffProcessor:
    def __init__(self, dat, dst):
        self.dat = dat

        self.md_info = {
            'step_per_image': 100,
            'time_step': 0.001,  # ps unit
        }

        self.trunc_ratio = (0.1, 0.7)
        self.dst = dst

    def run(self):
        loader = MSDDataLoader(self.dat, self.md_info, self.trunc_ratio)
        data = loader.get_data()

        plotter = DiffCoeffPlotter()
        E_a, D_0 = plotter.plot(data, self.dst)

        return data, E_a, D_0


class MSDDataLoader:
    def __init__(self, dat, md_info, trunc_ratio):
        self.md_info = md_info
        self.trunc_ratio = trunc_ratio
        self.dat = dat

    def get_data(self):
        '''
        Read all paths, and get diffusion coefficient.
        '''
        data = {}
        for temp, path in self.dat:
            if data.get(temp) is None:
                data[temp] = []
            x, y = self.read_msd(path, self.md_info)
            diff_coeff = self._get_diff_coeff(x, y, self.trunc_ratio)
            diff_coeff = diff_coeff.item()
            data[temp].append(diff_coeff)

        return data

    @staticmethod
    def _get_diff_coeff(x, y, trunc_ratio):
        '''
        From x, y, get diffusion coefficient.
        '''
        trunc_init, trunc_end = trunc_ratio
        x_init, x_end = int(len(x) * trunc_init), int(len(x) * trunc_end)
        x_trunc = x[x_init:x_end]
        y_init, y_end = int(len(y) * trunc_init), int(len(y) * trunc_end)
        y_trunc = y[y_init:y_end]
        A = np.vstack([x_trunc, np.ones(len(x_trunc))]).T
        slope, _ = np.linalg.lstsq(A, y_trunc, rcond=None)[0]
        diff_coeff = slope * Constants.A2_TO_CM2 / Constants.PS_TO_S / (2 * Constants.DIM)  # cm^2/s

        return diff_coeff

    @staticmethod
    def read_msd(path_dat, md_info):
        '''
        Read msd_avg.dat file and return x, y.
        '''
        dat = np.loadtxt(path_dat)
        md_step, _, _ = dat[:, 0], dat[:, 1], dat[:, 2]
        # msd_avg_SiF, _ = dat[:, 3], dat[:, 4]
        msd_avg_HF, _ = dat[:, 5], dat[:, 6]
        # msd_avg_others, _ = dat[:, 7], dat[:, 8]

        x = [i * md_info['step_per_image'] * md_info['time_step'] for i in md_step]
        y = msd_avg_HF

        return x, y


class DiffCoeffPlotter:
    @staticmethod
    def get_errorbar(x_data, y_data):
        '''
        Get errorbar data.
        '''
        errorbar_dict = {}
        for x, y in zip(x_data, y_data):
            if errorbar_dict.get(x) is None:
                errorbar_dict[x] = []
            errorbar_dict[x].append(y)
        x_err, y_mean, y_err = [], [], []
        for x, y_list in errorbar_dict.items():
            x_err.append(x)
            y_mean.append((np.max(y_list) + np.min(y_list)) / 2)
            y_err.append((np.max(y_list) - np.min(y_list)) / 2)
        return x_err, y_mean, y_err

    def plot(self, data, dst):
        '''
        Plot data.
        '''
        color = '#0c74b2'
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=(12, 6))

        x, y = [], []
        for (temp, diff_coeff) in data.items():
            for d in diff_coeff:
                if d < 0:
                    continue

                x.append(1/(Constants.KB*temp))
                y.append(np.log(d))
        x, y = np.array(x), np.array(y)

        # add trend line
        m, b = np.polyfit(x, y, 1)

        y_pred = m * x + b
        E_a = float(-m)
        D_0 = float(np.exp(b))
        label = f'$E_a$ = {E_a:.3f} eV\n'
        label += f'$D_0$ = {D_0:.2e} $cm^2/s$'

        prop_dict_line = {
            'linestyle': '--',
            'c': color,
        }
        ax.plot(x, y_pred, **prop_dict_line)

        prop_dict_errorbar = {
            'color': color,
            'fmt': '^',
            'capsize': 5,
        }
        x_err, y_mean, y_err = self.get_errorbar(x, y)

        eb = ax.errorbar(x_err, y_mean, yerr=y_err, **prop_dict_errorbar)
        eb[-1][0].set_linestyle('-')

        self.set_xy_axis_info(ax)
        self.add_xticks_temperature_in_units_of_K(ax)
        self.add_yticks_diff_coeff_in_power_of_ten(ax)

        fig.tight_layout()
        fig.savefig(dst)

        return E_a, D_0

    @staticmethod
    def set_xy_axis_info(ax):
        ax.set_xlabel('$1/(k_{B}T)$')
        ax.set_ylabel('$ln(D)$')
        # ax.legend(loc='center left', bbox_to_anchor=(1.20, 0.5), fontsize=14)
        y1, y2 = ax.get_ylim()
        y1, y2 = math.floor(np.log10(np.exp(y1))), math.ceil(np.log10(np.exp(y2)))
        ax.set_ylim(np.log(10**y1), np.log(10**y2))

    @staticmethod
    def add_xticks_temperature_in_units_of_K(ax):
        x1, x2 = ax.get_xlim()
        x1, x2 = 1/(Constants.KB*x2), 1/(Constants.KB*x1)
        x1, x2 = math.ceil(x1/50)*50, math.floor(x2/50)*50+1
        labels = np.arange(x1, x2, 50)
        xticks = 1/(Constants.KB*labels)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks, labels=labels)
        ax2.set_xlabel('Temperature (K)')

    @staticmethod
    def add_yticks_diff_coeff_in_power_of_ten(ax):
        ax3 = ax.twinx()
        ax3.set_ylim(ax.get_ylim())
        y1, y2 = ax.get_ylim()
        y1, y2 = math.ceil(np.log10(np.exp(y1))), math.floor(np.log10(np.exp(y2)))
        yticks = np.arange(y1, y2+1)
        labels = np.array([f"$10^{{{i}}}$" for i in yticks])
        yticks = np.log(np.power(10.0, yticks))
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(labels)
        ax3.set_ylabel('Diffusion Coefficient ($cm^2/s$)')


def plot_diffusion_coeff(src, temp_list, n_repeat, gas_type):
    dst_yaml = os.path.join("result_diffusion", f"diffusion_{gas_type}.yaml")
    if os.path.exists(dst_yaml):
        return

    os.makedirs("result_diffusion", exist_ok=True)
    result = []
    for temp in temp_list:
        for n in range(n_repeat):
            if not os.path.exists(os.path.join(src, f"{temp}/{n}", "FINAL.coo")):
                continue

            path_dump = os.path.join(src, f"{temp}/{n}", "dump.lammps")
            dst_dat = os.path.join("result_diffusion", f"msd_{gas_type}_{temp}_{n}.dat")
            msd = MeanSquaredDisplacementPlotter('MD', path_dump, dst_dat, gas_type)
            msd.run()

            result.append((temp, dst_dat))
    dst_png = os.path.join("result_diffusion", f"diffusion_{gas_type}.png")
    processor = DiffCoeffProcessor(result, dst_png)
    data, E_a, D_0 = processor.run()

    result = {}
    data = {int(k): v for k, v in data.items()}
    result['MSD'] = data
    result['E_a'] = E_a
    result['D_0'] = D_0
    with open(dst_yaml, 'w') as f:
        yaml.dump(result, f)
