import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_data(path_diff_dat):
    with open(path_diff_dat, 'r') as f:
        dat = yaml.load(f, Loader=yaml.FullLoader)

    return dat


def main():
    if len(sys.argv) != 2:
        print('Usage: python get_Eb_D0.py diff_dat.yaml')
        sys.exit()
    path_diff_dat = sys.argv[1]
    dat = read_data(path_diff_dat)
    kB = 8.617E-05  # eV/K

    x = []
    y = []
    for (temp, diff_coeff) in dat.items():
        for d in diff_coeff:
            x.append(1/(kB*temp))
            y.append(np.log(d))
    x = np.array(x)
    y = np.array(y)

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))
    prop_dict_scatter = {
        's': 100,
        'color': 'black',
    }
    ax.scatter(x, y, **prop_dict_scatter)

    # add trend line
    m, b = np.polyfit(x, y, 1)

    y_pred = m * x + b
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)

    title = f'$E_a$ = {-m:.3f} eV, '
    title += f'$D_0$ = {np.exp(b):.2e} $cm^2/s$'
    ax.set_title(title, font='Arial')
    prop_dict_line = {
        'linestyle': '--',
        'color': 'grey',
        'label': f'$R^2$ = {r_squared:.4f}',
    }
    ax.plot(x, y_pred, **prop_dict_line)

    ax.set_xlabel('$1/(k_{B}T)$')
    ax.set_ylabel('$ln(D)$')
    ax.legend(loc='upper right')

    x1, x2 = ax.get_xlim()
    x1, x2 = 1/(kB*x2), 1/(kB*x1)
    x1, x2 = math.ceil(x1/50)*50, math.floor(x2/50)*50+1
    labels = np.arange(x1, x2, 50)
    xticks = 1/(kB*labels)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xticks, labels=labels)
    ax2.set_xlabel('Temperature (K)')

    fig.tight_layout()
    fig.savefig('Eb_D0.png')


if __name__ == '__main__':
    main()
