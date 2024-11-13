import os
import sys
import math
from itertools import cycle

import yaml
import numpy as np
import matplotlib.pyplot as plt


KB = 8.617E-05  # eV/K
DIM = 2
A2_TO_CM2 = 1E-16
PS_TO_S = 1E-12


def get_diff_coeff(x, y, **fit_info):
    '''
    From x, y, get diffusion coefficient.
    '''
    trunc_init, trunc_end = fit_info['trunc_ratio']
    x_init, x_end = int(len(x) * trunc_init), int(len(x) * trunc_end)
    x_trunc = x[x_init:x_end]
    y_init, y_end = int(len(y) * trunc_init), int(len(y) * trunc_end)
    y_trunc = y[y_init:y_end]
    A = np.vstack([x_trunc, np.ones(len(x_trunc))]).T
    slope, _ = np.linalg.lstsq(A, y_trunc, rcond=None)[0]
    diff_coeff = slope * A2_TO_CM2 / PS_TO_S / (2 * DIM)  # cm^2/s

    return diff_coeff


def read_msd(path_dat, md_info, fit_info):
    '''
    Read msd_avg.dat file and return x, y.
    '''
    dat = np.loadtxt(path_dat)
    md_step, msd_avg, _ = dat[:, 0], dat[:, 1], dat[:, 2]
    # msd_avg_SiF, _ = dat[:, 3], dat[:, 4]
    msd_avg_HF, _ = dat[:, 5], dat[:, 6]
    # msd_avg_others, _ = dat[:, 7], dat[:, 8]

    x = [i * md_info['step_per_image'] * md_info['time_step'] for i in md_step]
    y = msd_avg_HF

    return x, y


def get_data(path_yaml, md_info, fit_info):
    '''
    Read all paths, and get diffusion coefficient.
    '''
    with open(path_yaml, 'r') as f:
        path_dict = yaml.safe_load(f)

    data = {}
    for cal_type in path_dict.keys():
        if data.get(cal_type) is None:
            data[cal_type] = {}

        for temp, path in path_dict[cal_type].items():
            if data[cal_type].get(temp) is None:
                data[cal_type][temp] = []

            folder_list = [
                f"{path}/{i}/msd/msd_avg.dat" for i in os.listdir(path)
                if i.isdigit()
            ]

            for path in folder_list:
                md_info_copy = md_info.copy()
                if cal_type == 'DFT':
                    md_info_copy['step_per_image'] = 1
                x, y = read_msd(path, md_info_copy, fit_info)
                diff_coeff = get_diff_coeff(x, y, **fit_info)
                diff_coeff = diff_coeff.item()

                data[cal_type][temp].append(diff_coeff)

    with open('diff_coeff.yaml', 'w') as f:
        yaml.dump(data, f)

    return data


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


def plot(data):
    '''
    Plot data.
    '''
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12, 6))
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(color_list)

    x_shift = np.linspace(-1, 1, len(data))

    for (cal_type, dat), color, x_s in zip(data.items(), color_cycle, x_shift):
        x, y = [], []
        for (temp, diff_coeff) in dat.items():
            for d in diff_coeff:
                x.append(1/(KB*temp))
                y.append(np.log(d))
        x, y = np.array(x), np.array(y)

        # add trend line
        m, b = np.polyfit(x, y, 1)

        y_pred = m * x + b
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_tot)

        label = f'{cal_type}\n'
        label += f'$E_a$ = {-m:.3f} eV\n'
        label += f'$D_0$ = {np.exp(b):.2e} $cm^2/s$'

        prop_dict_scatter = {
            's': 100,
            'color': color,
            'label': label,
            'alpha': 0.5,
        }

        prop_dict_line = {
            'linestyle': '--',
            'color': color,
            # 'label': f'$R^2$ = {r_squared:.4f}',
        }

        ax.scatter(x + x_s, y, **prop_dict_scatter)

        if cal_type == 'DFT':
            ax.plot(x, y_pred, **prop_dict_line)
            continue

        x_err, y_mean, y_err = get_errorbar(x, y, x_s)
        ax.errorbar(x_err, y_mean, yerr=y_err, color=color, fmt='^')

    ax.set_xlabel('$1/(k_{B}T)$')
    ax.set_ylabel('$ln(D)$')
    ax.legend(loc='center left', bbox_to_anchor=(1.20, 0.5), fontsize=14)

    x1, x2 = ax.get_xlim()
    x1, x2 = 1/(KB*x2), 1/(KB*x1)
    x1, x2 = math.ceil(x1/50)*50, math.floor(x2/50)*50+1
    labels = np.arange(x1, x2, 50)
    xticks = 1/(KB*labels)
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

    fig.tight_layout()
    fig.savefig(f'result.png')


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot.py path_dict.yaml")
        sys.exit(1)

    path_yaml = sys.argv[1]

    md_info = {
        'step_per_image': 100,
        'time_step': 0.001,  # ps unit
    }

    fit_info = {
        'trunc_ratio': (0.1, 0.7),
        # 'plot_options': {
        #     'linestyle': '--',
        # },
        # 'color': None,
    }

    data = get_data(path_yaml, md_info, fit_info)
    plot(data)


if __name__ == '__main__':
    main()
