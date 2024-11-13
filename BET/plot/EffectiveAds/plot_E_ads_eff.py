import sys

import matplotlib.pyplot as plt
import numpy as np


def read_data(path_energy_dat):
    with open(path_energy_dat, 'r') as f:
        lines = f.readlines()

        energy = []
        for line in lines:
            if line.startswith('#'):
                continue
            energy.append(-float(line.split()[2]))

    return np.array(energy)


def get_E_eff(E_ads, temp_range):
    kB = 8.617333262145e-5  # eV/K
    E_eff = []

    for T in temp_range:
        value = kB*T*np.log(np.sum(np.exp(E_ads/(kB*T)))/len(E_ads))
        E_eff.append(value)

    return E_eff


def plot(temp_range, E_eff):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    ax.plot(temp_range, E_eff)

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Effective adsorption energy (eV)')

    fig.tight_layout()
    fig.savefig('E_ads_eff.png')


def save_data(temp_range, E_eff):
    with open('dat', 'w') as f:
        for T, E in zip(temp_range, E_eff):
            f.write(f'{T} {E}\n')


def main():
    if len(sys.argv) != 2:
        print('Usage: python E_ads_eff_plot.py energy.dat')
        sys.exit(1)

    path_energy_dat = sys.argv[1]
    # temp_range = np.linspace(100, 1000, 30)
    temp_range = np.arange(100, 1000)

    energy = read_data(path_energy_dat)
    E_eff = get_E_eff(energy, temp_range)
    save_data(temp_range, E_eff)
    plot(temp_range, E_eff)


if __name__ == '__main__':
    main()
