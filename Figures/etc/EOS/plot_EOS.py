import os
import pickle
# from itertools import cycle

import matplotlib.pyplot as plt
from ase.eos import EquationOfState as EOS
from ase.io import read


def get_data_GNN(path_GNN):
    file_list = [
        f"{path_GNN}/{i}/thermo.dat" for i in os.listdir(path_GNN)
        if os.path.isdir(f"{path_GNN}/{i}") and 'v_' in i
    ]
    file_list = sorted(file_list, key=lambda x: float(x.split('/')[-2][2:]))

    E_list = []
    V_list = []

    for i in file_list:
        with open(i, "r") as f:
            lines = f.readlines()
            _, E, V, *_ = lines[-1].split()
            E_list.append(float(E))
            V_list.append(float(V))

    idx_E_min = E_list.index(min(E_list))
    atoms = read(file_list[idx_E_min].replace('thermo.dat', 'POSCAR_relaxed'))
    cell_params = atoms.cell.cellpar()

    return V_list, E_list, cell_params


def get_data_DFT(path_DFT):
    file_list = [
        f"{path_DFT}/{i}/OUTCAR" for i in os.listdir(path_DFT)
        if os.path.isdir(f"{path_DFT}/{i}") and 'v_' in i
    ]
    file_list = sorted(file_list, key=lambda x: float(x.split('/')[-2][2:]))

    E_list = []
    V_list = []

    for i in file_list:
        with open(i, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if "volume of cell" in line:
                    V = line.split()[-1]
                if "free  " in line:
                    E = line.split()[-2]
            E_list.append(float(E))
            V_list.append(float(V))

    idx_E_min = E_list.index(min(E_list))
    atoms = read(file_list[idx_E_min].replace('thermo.dat', 'POSCAR_relaxed'))
    cell_params = atoms.cell.cellpar()

    return V_list, E_list, cell_params


def plot_EOS(eos, ax, relative_energy=False, **prop_dict):
    plotdata = eos.getplotdata()
    eos_string, e0, v0, B, x, y, v, e = plotdata

    color = prop_dict.get('color')
    label = prop_dict.get('label')
    marker = prop_dict.get('marker')

    if relative_energy:
        y = [i - e0 for i in y]
        e = [i - e0 for i in e]
    ax.plot(x, y, ls='-', color=color, zorder=0)
    ax.scatter(v, e,
               marker=marker,
               edgecolor=color,
               label=label,
               s=10,
               facecolor='white',
               zorder=1
               )


def plot(relative_energy=False, **dat):
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=(3.5, 5))

    def convert_ax2(ax1):
        y1, y2 = ax1.get_ylim()
        natoms = 24
        ax2.set_ylim(y1 / natoms, y2 / natoms)
        ax2.figure.canvas.draw()

    ax2 = ax.twinx()
    ax.callbacks.connect("ylim_changed", convert_ax2)

    for k, (V, E, cell_params, style) in dat.items():
        color, marker = style
        eos = EOS(V, E)
        a, b, c, *_ = cell_params
        plot_EOS(eos,
                 ax,
                 color=color,
                 marker=marker,
                 label=k,
                 relative_energy=relative_energy)
        label = None
        # label = (f"V0: {eos.v0:.2f}"
        #          r" $\AA^3$"
        #          "\n"
        #          f"{a:.4f}"
        #          r"$\AA$"
        #          f" {b:.4f}"
        #          r"$\AA$"
        #          f" {c:.4f}"
        #          r"$\AA$")
        ax.axvline(eos.v0, color=color, ls=':', label=label, alpha=0.5)
        # label = f"E0 ({k}) = {eos.e0:.2f} eV"
        e0 = 0.0 if relative_energy else eos.e0
        # label = f"E0: {e0:.2f} eV"
        label = None
        ax.axhline(e0, color=color, ls='--', label=label, alpha=0.5)

    ax.set_xlabel(r"Volume ($\AA^3$)")
    ax.set_ylabel("Energy (eV)")
    # ax.legend(loc='center left', bbox_to_anchor=(1.3, 0.5), fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10)
    ax2.set_ylabel("(eV/atom)")

    fig.tight_layout()
    fig.savefig('EOS.png', dpi=200)


class StyleGenerator:
    @staticmethod
    def pick_style():
        color_list = ['#0c74b2',
                      '#D76224',
                      '#1BA077',
                      '#CA7AAA',
                      '#E8A125']
        marker_list = ['o',
                       's',
                       'D',
                       'v',
                       '^',
                       '<',
                       '>',
                       'p',
                       'P',
                       '*',
                       'h',
                       'H',
                       '+',
                       'x',
                       'X',
                       '|',
                       '_']
        style_list = [(color, marker)
                      for marker in marker_list
                      for color in color_list]

        for style in style_list:
            yield style

    @staticmethod
    def pick_color_from_color_cycle():
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for color in color_list:
            yield color

    @staticmethod
    def pick_color_from_dict(cal_type):
        color_dict = {
                'DFT': '#0c74b2',
                'chgTot': '#D76224',
                'vanilla': '#1BA077',
                }

        for key, value in color_dict.items():
            if key in cal_type:
                return value

        raise ValueError(f'Unknown cal_type: {cal_type}')


class DataLoader:
    def run(self, path_save='data.pkl'):
        if os.path.exists(path_save):
            with open(path_save, 'rb') as f:
                dat = pickle.load(f)
                return dat

        path_dict = {
            'DFT': "../DFT_newpot/",
            'chgTot': "../chgTot/",
            'high_conc_hf_large_baseline': "../high_conc_hf_large_baseline/",
            'high_conc_hf_large_vanilla': "../high_conc_hf_large_vanilla/",
            # 'high_conc_hf_small_baseline': "../high_conc_hf_small_baseline/",
            'high_conc_hf_small_vanilla': "../high_conc_hf_small_vanilla/",
        }

        dat = {}
        gen_style = StyleGenerator.pick_style()
        for pot, path in path_dict.items():
            if 'DFT' in pot:
                read_func = get_data_DFT
            else:
                read_func = get_data_GNN
            style = next(gen_style)
            dat[pot] = (*read_func(path), style)

        with open(path_save, 'wb') as f:
            pickle.dump(dat, f)

            return dat

def main():
    dl = DataLoader()
    dat = dl.run()

    relative_energy = True
    plot(relative_energy=relative_energy, **dat)


if __name__ == "__main__":
    main()
