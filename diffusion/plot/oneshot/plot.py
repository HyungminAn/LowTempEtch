import matplotlib.pyplot as plt
import numpy as np

from ase.io import read


def read_data_gnn(path_gnn):
    path_gnn_list = [f'{path_gnn}/{i}/thermo.dat' for i in range(101)]
    data_gnn = {}
    for path in path_gnn_list:
        with open(path, 'r') as f:
            energy = float(f.readlines()[-1].split()[1])
            data_gnn[int(path.split('/')[-2])] = energy

    return data_gnn


def read_data_dft(path_dft):
    path_dft_list = [f'{path_dft}/{i}/OUTCAR' for i in range(101)]
    data_dft = {}
    for path in path_dft_list:
        with open(path, 'r') as f:
            for line in f.readlines():
                if 'free  ' in line:
                    energy = float(line.split()[-2])
                    break
        data_dft[int(path.split('/')[-2])] = energy

    return data_dft


def read_nions(path_gnn):
    path_gnn = f'{path_gnn}/0/POSCAR'
    nions = len(read(path_gnn))

    return nions


def plot(data_gnn, data_dft, nions):
    x = np.array([i for i in data_gnn.keys()]) * 0.1
    y_gnn = np.array([i for i in data_gnn.values()])
    y_dft = np.array([i for i in data_dft.values()])
    y_diff = (y_gnn - y_dft) / nions * 1000  # meV/atom

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    ax, ax_diff = axes
    prop_dict_GNN = {
        'color': 'blue',
        'marker': 'o',
        'label': 'GNN',
        'linestyle': '-'
    }
    prop_dict_DFT = {
        'color': 'black',
        'marker': 'x',
        'label': 'DFT',
        'linestyle': '--'
    }
    ax.plot(x, y_gnn, **prop_dict_GNN)
    ax.plot(x, y_dft, **prop_dict_DFT)

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Energy (eV)')

    ax.legend()

    prop_dict_diff = {
        'color': 'red',
        'marker': 'x',
        'label': 'GNN - DFT',
        'linestyle': '--'
    }
    ax_diff.plot(x, y_diff, **prop_dict_diff)
    ax_diff.set_xlabel('Time (ps)')
    ax_diff.set_ylabel('Energy Difference (meV/atom)')
    ax_diff.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_diff.legend()

    fig.tight_layout()
    fig.savefig('energy.png')


def main():
    path_gnn = \
    "/data2/andynn/LowTempEtch/04_diffusion/AFS/chgTot/oneshotDFT/SmallCell_IF5_1ML_HF_1_400K_10ps/chgTot"
    path_dft = \
    "/data2/andynn/LowTempEtch/04_diffusion/AFS/chgTot/oneshotDFT/SmallCell_IF5_1ML_HF_1_400K_10ps/DFT"
    data_gnn = read_data_gnn(path_gnn)
    data_dft = read_data_dft(path_dft)
    nions = read_nions(path_gnn)
    plot(data_gnn, data_dft, nions)


if __name__ == "__main__":
    main()
