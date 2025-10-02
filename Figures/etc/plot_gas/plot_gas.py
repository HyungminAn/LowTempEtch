import os

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import mean_absolute_error as MAE

# from ase.io import read
from adjustText import adjust_text


def filter_XF5_and_HF(string):
    if ("F5" in string and "C2" not in string) or "HF" in string:
        return True
    else:
        return False


class DataReader:
    @staticmethod
    def get(path_DFT, path_GNN):
        E_GNN = DataReader._get_data_GNN(path_GNN, filterfunc=filter_XF5_and_HF)
        E_DFT = DataReader._get_data_DFT(path_DFT, filterfunc=filter_XF5_and_HF)
        return E_GNN, E_DFT

    @staticmethod
    def _get_data_GNN(path_GNN, filterfunc=None):
        if filterfunc is None:
            file_list = [
                f"{path_GNN}/{i}/thermo.dat" for i in os.listdir(path_GNN)
                if os.path.isdir(f"{path_GNN}/{i}")
            ]
        else:
            file_list = [
                f"{path_GNN}/{i}/thermo.dat" for i in os.listdir(path_GNN)
                if os.path.isdir(f"{path_GNN}/{i}") and filterfunc(i)
            ]
        E_dict = {}

        for i in file_list:
            mol_name = i.split('/')[-2]
            with open(i, "r") as f:
                lines = f.readlines()
                _, E, *_ = lines[-1].split()
                E_dict[mol_name] = float(E)

        return E_dict

    @staticmethod
    def _get_data_DFT(path_DFT, filterfunc=None):
        if filterfunc is None:
            file_list = [
                f"{path_DFT}/{i}/OUTCAR" for i in os.listdir(path_DFT)
                if os.path.isdir(f"{path_DFT}/{i}")
            ]
        else:
            file_list = [
                f"{path_DFT}/{i}/OUTCAR" for i in os.listdir(path_DFT)
                if os.path.isdir(f"{path_DFT}/{i}") and filterfunc(i)
            ]

        E_dict = {}

        for i in file_list:
            mol_name = i.split('/')[-2]
            with open(i, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "free  " in line:
                        E = line.split()[-2]
                E_dict[mol_name] = float(E)

        return E_dict


class Plotter:
    @staticmethod
    def plot(data):
        E_GNN, E_DFT = data
        keys = [i for i in E_GNN.keys()]
        E_GNN = np.array([E_GNN[key] for key in keys])
        E_DFT = np.array([E_DFT[key] for key in keys])
        dE = np.abs(E_GNN - E_DFT)

        plt.rcParams.update({'font.size': 18})
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        ax, ax_bar = axes

        Plotter._draw_parity_plot(E_GNN, E_DFT, dE, ax, keys)
        Plotter._draw_bar_plot(E_GNN, E_DFT, dE, ax_bar, keys)

        fig.tight_layout()
        fig.savefig('E_gas.png')

    @staticmethod
    def _draw_parity_plot(E_GNN, E_DFT, dE, ax, keys):
        ax.scatter(E_DFT, E_GNN, marker='o', color='#0c74b2')

        ax.set_aspect('equal')
        ax.set_xlabel("$E_{DFT}$ (eV)")
        ax.set_ylabel("$E_{GNN}$ (eV)")

        E_cutoff = 1.0  # eV
        # shift = 0.5
        # for (mol, x, y, E) in zip(keys, E_DFT, E_GNN, dE):
        #     if np.abs(E) > E_cutoff:
        #         text = f"{mol}, {E:.2f} eV"
        #         ax.text(x, y, text, fontsize=8)

        ax.axline((0, 0), slope=1, color='k', linestyle='--', alpha=0.5)
        ax.axline((E_cutoff, 0), slope=1, color='grey', linestyle='--', alpha=0.5)
        ax.axline((-E_cutoff, 0), slope=1, color='grey', linestyle='--', alpha=0.5)

        x, y = E_GNN, E_DFT
        texts = []
        for i, txt in enumerate(keys):
            txt_position = (x[i], y[i])
            texts.append(ax.text(txt_position[0], txt_position[1], txt, fontsize=18))

        adjust_text_props = {
            'arrowprops': dict(arrowstyle='->', color='black', lw=0.5),
            'expand': (1, 1),
            'ax': ax,
        }
        adjust_text(texts, x, y, **adjust_text_props)

    @staticmethod
    def _draw_bar_plot(E_GNN, E_DFT, dE, ax_bar, keys):
        idx = np.argsort(dE)
        x_bar = np.array(keys)[idx]
        y_bar = dE[idx]
        bar_plot = ax_bar.barh(x_bar, y_bar, color='#0c74b2')
        ax_bar.bar_label(bar_plot, labels=[f"{i:.2f}" for i in y_bar], padding=3)
        ax_bar.set_xlabel("Error in gas energy (eV)")
        ax_bar.set_xlim(None, 1.3)

        # text = os.getcwd().split('/')[-1]
        # prop_text = {
        #         'ha': 'bottom',
        #         'va': 'right',
        #         'fontsize': 20,
        #         'transform': ax_bar.transAxes,
        #         'bbox': dict(facecolor='wheat', alpha=0.5),
        #         }
        # ax.text(0.05, 0.95, text, **prop_text)

        # mae = MAE(E_DFT, E_GNN)
        # text_mae = f"MAE = {mae:.2f} eV"
        # prop_text['ha'] = 'right'
        # prop_text['va'] = 'bottom'
        # ax.text(0.95, 0.05, text_mae, **prop_text)


def main():
    path_GNN = "../chgTot/"
    path_vasp = "../DFT/"

    data = DataReader.get(path_vasp, path_GNN)

    Plotter.plot(data)


if __name__ == "__main__":
    main()
