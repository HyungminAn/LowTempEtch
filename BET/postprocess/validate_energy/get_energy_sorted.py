import os

import numpy as np
import pandas as pd


def main():
    src = "/data2/andynn/LowTempEtch/07_Equil_LayerThickness"
    path_bareSurf = f"{src}/03_bareSurface/results"
    path_ADDIonHF = f"{src}/05_ADDIonHF/results"
    path_HFonADDI = f"{src}/06_HFonADDI/results"
    path_ADDIonADDI = f"{src}/07_ADDIonADDI/results"
    check_list = {
        (path_bareSurf, "bareSurface"),
        (path_ADDIonHF, "ADDIonHF"),
        (path_HFonADDI, "HFonADDI"),
        (path_ADDIonADDI, "ADDIonADDI"),
    }

    gas_list = [
        i for i in os.listdir(path_bareSurf)
        if os.path.isdir(os.path.join(path_bareSurf, i))
    ]

    energy_dict = {
        'gas': [],
        'label': [],
        'index': [],
        'energy': [],
    }

    for gas in gas_list:
        for path, label in check_list:
            data = np.loadtxt(f"{path}/{gas}/plot/EffectiveAds/energy.dat", usecols=(2))
            for i, v in enumerate(data.flatten()):
                energy_dict['gas'].append(gas)
                energy_dict['label'].append(label)
                energy_dict['index'].append(i)
                energy_dict['energy'].append(v)
            v_mean = np.mean(data)
            v_min = np.min(data)
            v_max = np.max(data)
            v_range = v_max - v_min
            result = [
                (v_mean, "mean"),
                (v_min, "min"),
                (v_max, "max"),
                (v_range, "range"),
            ]
            for v, v_type in result:
                energy_dict['gas'].append(gas)
                energy_dict['label'].append(label)
                energy_dict['index'].append(v_type)
                energy_dict['energy'].append(v)
            print(f"{gas} {label} Done")

    df = pd.DataFrame(energy_dict)
    df.to_csv("summary.csv")


if __name__ == "__main__":
    main()
