import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DiffusionCoefficientPlotter():
    KB = 8.617E-5  # eV/K
    DIM = 2
    A2_TO_CM2 = 1E-16
    PS_TO_S = 1E-12

    def __init__(self, dst, path_dict, md_info=None, fit_info=None):
        self.dst = Path(dst)
        self.path_dict = path_dict
        self.md_info = md_info or {
            'step_per_image': 100,
            'time_step': 0.001,  # ps unit
        }
        self.fit_info = fit_info or {
            'trunc_ratio': (0.1, 0.7),
        }
        self.data = None

    def run(self):
        self._get_data()
        self._plot()

    def _get_diff_coeff(self, x, y):
        trunc_init, trunc_end = self.fit_info['trunc_ratio']
        x_init, x_end = int(len(x) * trunc_init), int(len(x) * trunc_end)
        x_trunc = x[x_init:x_end]
        y_trunc = y[x_init:x_end]
        A = np.vstack([x_trunc, np.ones(len(x_trunc))]).T
        slope, _ = np.linalg.lstsq(A, y_trunc, rcond=None)[0]
        diff_coeff = slope * self.A2_TO_CM2 / self.PS_TO_S / (2 * self.DIM)  # cm^2/s
        return diff_coeff

    def _read_msd(self, path_dat, md_info):
        dat = np.loadtxt(path_dat)
        md_step, msd_avg_HF = dat[:, 0], dat[:, 5]
        x = [i * md_info['step_per_image'] * md_info['time_step'] for i in md_step]
        return x, msd_avg_HF

    def _get_data(self):
        data = {}
        for temp, repeats in self.path_dict.items():
            data[temp] = []

            for path in repeats:
                md_info_copy = self.md_info.copy()
                x, y = self._read_msd(path, md_info_copy)
                diff_coeff = self._get_diff_coeff(x, y)
                data[temp].append(diff_coeff)

        self.data = data

    def _get_errorbar(self, x_data, y_data):
        errorbar_dict = {}
        for x, y in zip(x_data, y_data):
            if x not in errorbar_dict:
                errorbar_dict[x] = []
            errorbar_dict[x].append(y)

        x_err, y_mean, y_err = [], [], []
        for x, y_list in errorbar_dict.items():
            x_err.append(x)
            y_mean.append((np.max(y_list) + np.min(y_list)) / 2)
            y_err.append((np.max(y_list) - np.min(y_list)) / 2)
        return x_err, y_mean, y_err

    def _plot(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=(12, 6))

        x, y = [], []
        for temp, diff_coeff in self.data.items():
            for d in diff_coeff:
                _x = 1/(self.KB*temp)
                _y = np.log(d)
                x.append(_x)
                y.append(_y)
                print(f"***** {temp:3d} {_y:.4e}")
        x, y = np.array(x), np.array(y)

        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b

        self.diff_barrier = -m
        self.diff_coeff = np.exp(b)

        label = f'$E_a$ = {self.diff_barrier:.3f} eV\n'
        label += f'$D_0$ = {self.diff_coeff:.2e} $cm^2/s$'

        ax.scatter(x, y, s=100, label=label, alpha=0.5)
        ax.plot(x, y_pred, linestyle='--')

        x_err, y_mean, y_err = self._get_errorbar(x, y)
        ax.errorbar(x_err, y_mean, yerr=y_err, fmt='^')

        self._set_axes(ax)
        fig.tight_layout()
        fig.savefig(self.dst/'result.png')

    def _set_axes(self, ax):
        ax.set_xlabel('$1/(k_{B}T)$')
        ax.set_ylabel('$ln(D)$')
        ax.legend(loc='center left', bbox_to_anchor=(1.20, 0.5), fontsize=14)

        x1, x2 = ax.get_xlim()
        x1, x2 = 1/(self.KB*x2), 1/(self.KB*x1)
        x1, x2 = math.ceil(x1/50)*50, math.floor(x2/50)*50+1
        labels = np.arange(x1, x2, 50)
        xticks = 1/(self.KB*labels)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks, labels=labels)
        ax2.set_xlabel('Temperature (K)')

        y1, y2 = ax.get_ylim()
        y1, y2 = math.floor(np.log10(np.exp(y1))), math.ceil(np.log10(np.exp(y2)))
        ax.set_ylim(np.log(10**y1), np.log(10**y2))

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
