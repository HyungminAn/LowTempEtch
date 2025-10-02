import os
import sys
import math
from dataclasses import dataclass

import yaml
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Constants:
    KB = 8.617E-05  # eV/K
    DIM = 2
    A2_TO_CM2 = 1E-16
    PS_TO_S = 1E-12


class DiffCoeffProcessor:
    def __init__(self, path_yaml):
        self.path_yaml = path_yaml

        self.md_info = {
            'step_per_image': 100,
            'time_step': 0.001,  # ps unit
        }

        self.trunc_ratio = (0.1, 0.7)

    def run(self):
        loader = MSDDataLoader(self.path_yaml, self.md_info, self.trunc_ratio)
        data = loader.get_data()

        plotter = DiffCoeffPlotter()
        plotter.plot(data)


class MSDDataLoader:
    def __init__(self, path_yaml, md_info, trunc_ratio):
        self.md_info = md_info
        self.trunc_ratio = trunc_ratio
        with open(path_yaml, 'r') as f:
            self.path_dict = yaml.safe_load(f)

    def get_data(self, path_save='diff_coeff.yaml'):
        '''
        Read all paths, and get diffusion coefficient.
        '''
        if os.path.exists(path_save):
            with open(path_save, 'r') as f:
                data = yaml.safe_load(f)
            return data

        data = {}
        path_dict = self.path_dict

        for cal_type in path_dict.keys():
            if data.get(cal_type) is None:
                data[cal_type] = {}

            for temp, path in path_dict[cal_type].items():
                if data[cal_type].get(temp) is None:
                    data[cal_type][temp] = []

        for cal_type in path_dict.keys():
            for temp, path in path_dict[cal_type].items():
                folder_list = [f"{path}/{i}"
                               for i in os.listdir(path)
                               if i.isdigit()]
                if folder_list:
                    folder_list = [
                        f"{path}/{i}/msd/msd_avg.dat"
                        for i in os.listdir(path)
                        if i.isdigit()
                    ]
                else:
                    folder_list = [f"{path}/msd/msd_avg.dat"]

                for path in folder_list:
                    md_info_copy = self.md_info.copy()
                    if 'DFT' in cal_type:
                        md_info_copy['step_per_image'] = 1
                    x, y = self.read_msd(path, md_info_copy)
                    diff_coeff = self._get_diff_coeff(x, y, self.trunc_ratio)
                    diff_coeff = diff_coeff.item()
                    data[cal_type][temp].append(diff_coeff)

        with open(path_save, 'w') as f:
            yaml.dump(data, f)

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
    def get_errorbar(x_data, y_data, x_s):
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
            x_err.append(x + x_s)
            y_mean.append((np.max(y_list) + np.min(y_list)) / 2)
            y_err.append((np.max(y_list) - np.min(y_list)) / 2)
        return x_err, y_mean, y_err

    def _pick_color(self, cal_type):
        '''
        Assign colors for each calculation type.
        '''
        color_dict = {
            'AFS slab': '#4fb696',
            'AFS slab + IF$_5$ 1ML': '#ca7baa',
        }

        if cal_type in color_dict:
            return color_dict[cal_type]

        raise ValueError(f'Unknown cal_type: {cal_type}')

    def plot(self, data):
        '''
        Plot data with updated styles and layout.
        '''
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'Arial',
            })
        cm = 1 / 2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(3.5, 4.2))

        x_shift = np.linspace(-1, 1, len(data)) * 0.5  # Marker 간격 조정
        for (cal_type, dat), x_s in zip(data.items(), x_shift):
            color = self._pick_color(cal_type)
            x, y = self._get_xy(dat)

            x = x[1:]
            y = y[1:]

            # E_a, D_0 = self._get_slope_and_draw_trendline(ax, x, y, cal_type, color)

            label = f'{cal_type}'
            self._draw_errorbar(ax, x, y, x_s, cal_type, color, label)

        self.set_xy_axis_info(ax)
        self.add_xticks_temperature_in_units_of_K(ax)
        self.add_yticks_diff_coeff_in_power_of_ten(ax)

        fig.tight_layout()
        fig.savefig(f'result.png', dpi=200)
        fig.savefig(f'result.pdf', dpi=200)


    @staticmethod
    def _get_xy(dat):
        x, y = [], []
        for (temp, diff_coeff) in dat.items():
            for d in diff_coeff:
                if d < 0:
                    continue

                x.append(1/(Constants.KB*temp))
                y.append(np.log(d))
        x, y = np.array(x), np.array(y)
        return x, y

    @staticmethod
    def _get_slope_and_draw_trendline(ax, x, y, cal_type, color):
        is_slope_available = len(set(x)) > 2
        if not is_slope_available:
            return None, None

        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        E_a = -m
        D_0 = np.exp(b)
        # ss_tot = np.sum((y - np.mean(y))**2)
        # ss_res = np.sum((y - y_pred)**2)
        # r_squared = 1 - (ss_res / ss_tot)

        is_trendline_available = cal_type == 'DFT'
        if is_trendline_available:
            prop_dict_line = {
                'linestyle': '--',
                'color': color,
                # 'label': f'$R^2$ = {r_squared:.4f}',
            }
            ax.plot(x, y_pred, **prop_dict_line)
        print(f'{cal_type}: E_a = {E_a:.3f} eV, D_0 = {D_0:.2e} cm^2/s')
        return E_a, D_0

    def _draw_errorbar(self, ax, x, y, x_s, cal_type, color, label):
        '''
        Draw error bars with custom styles.
        '''
        prop_dict_errorbar = {
            'color': color,
            'label': label,
            'fmt': '^',  # Marker style
            'capsize': 3,
            'linestyle': '-',
            'markeredgecolor': 'black',
            'linewidth': 0,
            'elinewidth': 0.8,
            'markersize': 5
        }
        x_err, y_mean, y_err = self.get_errorbar(x, y, x_s)

        eb = ax.errorbar(x_err, y_mean, yerr=y_err, **prop_dict_errorbar)
        eb[-1][0].set_linestyle('-')

    @staticmethod
    def set_xy_axis_info(ax):
        # ax.set_xlabel('$1/(k_{B}T)$')
        ax.set_xlabel('1000/T (K$^{-1}$)')

        conv_factor = Constants.KB * 1000
        step = 0.5
        x1_in_kBT, x2_in_kBT = ax.get_xlim()
        x1_in_T, x2_in_T = conv_factor * x1_in_kBT, conv_factor * x2_in_kBT
        x1_in_T, x2_in_T = math.ceil(x1_in_T/step)*step, (math.floor(x2_in_T/step)+1)*step
        xticks = np.arange(x1_in_T, x2_in_T, step) / conv_factor
        labels = np.array([f"{i:.1f}" for i in np.arange(x1_in_T, x2_in_T, step)])
        ax.set_xticks(xticks, labels=labels)

        ax.set_ylabel('ln($D$)')
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, -0.2),
                  fontsize=10,
                  frameon=False,
                  )
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
        ax3.set_ylabel('Diffusion Coefficient (cm$^2$/s)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot.py path_dict.yaml")
        sys.exit(1)

    path_yaml = sys.argv[1]
    processor = DiffCoeffProcessor(path_yaml)
    processor.run()
