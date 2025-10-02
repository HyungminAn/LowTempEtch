import os
import sys
import yaml

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adsorption import plot_adsorption_energy
from diffusion import plot_diffusion_coeff


def check_diffusion_finished(src, temp_list, n_repeat, print_=False):
    path_diffusion = os.path.join(src, "run", "diffusion")
    total_count = len(temp_list) * n_repeat
    count = 0

    for temp in temp_list:
        for n in range(n_repeat):
            path = os.path.join(path_diffusion, f"{temp}/{n}", "FINAL.coo")
            if os.path.exists(path):
                count += 1

    if count == total_count:
        if print_:
            print(f"{path_diffusion:<80}\tfinished({count}/{total_count})")
        return True
    else:
        if print_:
            print(f"{path_diffusion:<80}\tnot finished({count}/{total_count})")
        return False


def convert_time(time):
    time = time.split(":")
    time = [int(t) for t in time]
    time = time[0] * 3600 + time[1] * 60 + time[2]
    return round(time / 3600, 2)


def check_md_time(src, temp_list, n_repeat, gas_type):
    path_diffusion = os.path.join(src, "run", "diffusion")
    time_list = []
    for temp in temp_list:
        for n in range(n_repeat):
            path = os.path.join(path_diffusion, f"{temp}/{n}", "log.lammps")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                lines = f.readlines()
            time = lines[-1].split()[-1]
            time = convert_time(time)
            time_list.append(time)
    time_list = np.array(time_list)
    t_min, t_avg, t_max = np.min(time_list), np.mean(time_list), np.max(time_list)
    line = (
            f"{gas_type:<10}\t"
            f"{t_min:.2f} h\t"
            f"{t_avg:.2f} h\t"
            f"{t_max:.2f} h"
            )
    print(line)
    return t_min, t_avg, t_max


def get_J_diff(E_ads_eff, D_0, E_a, T):
    kB = 8.617333262145e-5  # eV/K
    J_diff = D_0 * np.exp((E_ads_eff - E_a) / (kB * T))
    return J_diff


def print_results(data):
    T_target = 213  # K
    sorted_dict = {}
    for gas_type, src in data.items():
        path_result_ads = f"./result_adsorption/ads_{gas_type}.yaml"
        path_result_diff = f"./result_diffusion/diffusion_{gas_type}.yaml"
        if not os.path.exists(path_result_ads) or not os.path.exists(path_result_diff):
            continue

        with open(path_result_ads, "rb") as f:
            result_ads = yaml.load(f, Loader=yaml.FullLoader)
        with open(path_result_diff, "rb") as f:
            result_diff = yaml.load(f, Loader=yaml.FullLoader)

        E_ads_eff = result_ads["E_ads_eff"]
        D_0 = result_diff["D_0"]
        E_a = result_diff["E_a"]
        J_diff = get_J_diff(E_ads_eff, D_0, E_a, T_target)
        sorted_dict[gas_type] = (E_ads_eff, D_0, E_a, J_diff)

    sorted_list = sorted(sorted_dict.items(), key=lambda x: x[1][3], reverse=True)

    for (gas_type, (E_ads_eff, D_0, E_a, J_diff)) in sorted_list:
        line = (
                f"{gas_type:<10}\t"
                f"{E_ads_eff:5.2f}\teV\t"
                f"{D_0:5.2e}\tcm^2/s\t"
                f"{E_a:5.2f}\teV\t"
                f"{J_diff:5.2e}\tcm^2/s"
                )
        print(line)


def main():
    path_yaml = "path.yaml"
    with open(path_yaml, "r") as f:
        data = yaml.safe_load(f)

    temp_list = [250, 300, 350]
    n_repeat = 3
    for gas_type, src in data.items():
        is_diff_finished = check_diffusion_finished(src, temp_list, n_repeat, print_=True)
        if not is_diff_finished:
            continue

    for gas_type, src in data.items():
        is_diff_finished = check_diffusion_finished(src, temp_list, n_repeat, print_=False)
        if not is_diff_finished:
            continue
        check_md_time(src, temp_list, n_repeat, gas_type)

    for gas_type, src in data.items():
        plot_adsorption_energy(src)

    for gas_type, src in data.items():
        src_diff = os.path.join(src, "run", "diffusion")
        plot_diffusion_coeff(src_diff, temp_list, n_repeat, gas_type)

    print_results(data)


if __name__ == "__main__":
    main()
