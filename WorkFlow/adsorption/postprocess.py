import os
import sys
import yaml
from pathlib import Path

import numpy as np

# To import other scripts
script_directory = Path(__file__).parent.absolute()
sys.path.append(str(script_directory))

from utils.log import log_function_call  # noqa: E402
from check_disso import is_etchant_dissociated  # noqa: E402
from check_surf_reconst import is_surface_reconstructed  # noqa: E402


@log_function_call
def get_slab_E(output, **inputs):
    '''
    Get the energy of slab
    '''
    key = "additive"
    inputs[key]["paths"]["dst_1"] =\
        inputs[key]["paths"]["dst_1"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_2"] =\
        inputs[key]["paths"]["dst_2"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_3"] =\
        inputs[key]["paths"]["dst_3"] % {'dst': inputs['dst']}

    energy_min = 0

    path_src = inputs[key]["paths"]["dst_2"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    max_step = inputs["relax"]["max_steps"]
    for i in range(n_repeat):
        src = f'{path_src}/{i}'
        with open(f'{src}/thermo.dat', 'r') as f:
            last_line = f.readlines()[-1]

        step, energy, *_ = last_line.split()
        step = int(step)
        energy = float(energy)

        if step >= max_step:
            continue

        if energy <= energy_min:
            # i_save = i
            energy_save = energy

    output["E_slab"] = energy_save


@log_function_call
def check_etchant_dissociation(key, **inputs):
    '''
    classify results to physisorption and chemisorption,
    depending on whether the inserted etchant molecule has been dissociated.
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    path_slab = inputs[key]["paths"]["slab"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    idx_etchant_dict = {}
    for i in range(n_repeat):
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
            continue
        cond, idx_etchant = is_etchant_dissociated(src, path_slab)
        if cond:
            continue
        idx_etchant_dict[i] = idx_etchant

    return idx_etchant_dict


@log_function_call
def check_reconstruction(key, idx_etchant_dict, **inputs):
    '''
    Check whether there was a reconstruction during structure optimization
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    idx_reconst = {}
    for i, idx_etchant in idx_etchant_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
            continue

        if is_surface_reconstructed(src, idx_etchant):
            idx_reconst[i] = idx_etchant

    for k in idx_reconst.keys():
        idx_etchant_dict.pop(k, None)

    return idx_reconst


@log_function_call
def check_reconstruction_reverse(key, idx_etchant_dict, **inputs):
    '''
    check_reconstruction function in reverse_direction
    '''
    path_src = inputs[key]["paths"]["dst_3"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    idx_reconst = {}

    for i, idx_etchant in idx_etchant_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
            continue

        if is_surface_reconstructed(src, []):
            idx_reconst[i] = None

    for k in idx_reconst.keys():
        idx_etchant_dict.pop(k, None)

    return idx_reconst


@log_function_call
def split_results(output, **inputs):
    '''
        Split the results into two groups: chemisorption and physisorption
    '''
    key = "etchant"
    inputs[key]["paths"]["slab"] =\
        inputs[key]["paths"]["slab"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_1"] =\
        inputs[key]["paths"]["dst_1"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_2"] =\
        inputs[key]["paths"]["dst_2"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_3"] =\
        inputs[key]["paths"]["dst_3"] % {'dst': inputs['dst']}

    idx_etchant_dict = check_etchant_dissociation(key, **inputs)
    idx_reconst_dict = check_reconstruction(key, idx_etchant_dict, **inputs)

    exclude_dict = check_reconstruction_reverse(
            key, idx_reconst_dict, **inputs)
    phys_dict = {**idx_etchant_dict, **idx_reconst_dict}

    output["exclude_dict"] = exclude_dict
    output["phys_dict"] = phys_dict


def get_physisorption_ratio(key, output, **inputs):
    exclude_dict = output["exclude_dict"]
    phys_dict = output["phys_dict"]

    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    n_fail = len(exclude_dict)
    r_fail = n_fail / n_repeat

    n_phys = len(phys_dict)
    n_chem = n_repeat - n_fail - n_phys
    r_chem = n_chem / n_repeat
    r_phys = n_phys / n_repeat

    return r_phys, r_chem, r_fail


def read_energy_from_thermodat(src, max_step):
    with open(src, 'r') as f:
        step, E, *_ = f.readlines()[-1].split()
        step = int(step)
        E = float(E)
        if step >= max_step:
            return None

        return E


def save_adsorption_energy_data(name, E_ads_dict, dir_reconst):
    with open(f'{name}.dat', 'w') as f:
        for i, E_ads in E_ads_dict.items():
            line = f'{i:>10d} {E_ads:10.3f} '
            if i in dir_reconst:
                line += 'Reconst\n'
            else:
                line += 'Normal\n'
            f.write(line)


def get_adsorption_energy(name, key, output, **inputs):
    # Get energy of mol
    path_mol = inputs[key]["paths"]["mol"]
    path_mol_dat = '/'.join(path_mol.split('/')[:-1]) + '/log.lammps'
    with open(path_mol_dat, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'Energy initial' in line:
                E_mol = float(lines[idx+1].split()[-1])
                break

    src_no_reconst = inputs[key]["paths"]["dst_2"]
    src_reconst = inputs[key]["paths"]["dst_3"]
    dir_reconst = os.listdir(src_reconst) if os.path.isdir(src_reconst) else []
    dir_reconst = [int(i) for i in dir_reconst]
    max_step = inputs["relax"]["max_steps"]

    phys_dict = output["phys_dict"]

    # Get energy of slab + mol
    E_slabMol_dict = {}
    for i in phys_dict.keys():
        src = f'{src_no_reconst}/{i}/thermo.dat'
        E = read_energy_from_thermodat(src, max_step)
        if E is not None:
            E_slabMol_dict[i] = E

    # Get energy of slab
    E_ads_dict = {}
    for i, E_slabMol in E_slabMol_dict.items():
        if i in dir_reconst:
            src = f'{src_reconst}/{i}/thermo.dat'
            E_slab = read_energy_from_thermodat(src, max_step)
            if E_slab is None:
                continue
        else:
            E_slab = output["E_slab"]

        E_ref = E_mol + E_slab
        E_ads_dict[i] = E_slabMol - E_ref

    save_adsorption_energy_data(name, E_ads_dict, dir_reconst)

    return E_ads_dict


@log_function_call
def summarize_results(name, output, **inputs):
    '''
        summarize the results
    '''
    key = "etchant"
    r_phys, r_chem, r_fail = get_physisorption_ratio(key, output, **inputs)
    line = f'{name:>10s} {r_phys:7.2f} {r_chem:7.2f} {r_fail:7.2f} '

    E_ads_dict = get_adsorption_energy(name, key, output, **inputs)
    E_ads_idx_array = np.array([i for i in E_ads_dict.keys()])
    E_ads_array = np.array([i for i in E_ads_dict.values()])
    E_avg = np.average(E_ads_array)
    E_max = np.max(E_ads_array)
    idx_E_max = E_ads_idx_array[np.argmax(E_ads_array)]
    E_min = np.min(E_ads_array)
    idx_E_min = E_ads_idx_array[np.argmin(E_ads_array)]
    line += f'{E_avg:10.3f} {E_max:10.3f} {E_min:10.3f} '
    line += f'{idx_E_max:12d} {idx_E_min:12d}\n'
    print(line)

    return line


@log_function_call
def main():
    src = "/data2/andynn/LowTempEtch/10_BatchRun"
    folder_list = [
        f'{src}/{i}' for i in os.listdir(src) if os.path.isdir(f'{src}/{i}')]

    with open('result.txt', 'w') as fd1:
        line = f'{"name":>10s} {"r_phys":>7s} {"r_chem":>7s} {"r_fail":>7s} '
        line += f'{"E_avg":>10s} {"E_max":>10s} {"E_min":>10s} '
        line += f'{"idx_E_max":>12s} {"idx_E_min":>12s}\n'
        fd1.write(line)

        for folder in folder_list:
            path_inputs = f"{folder}/input.yaml"
            with open(path_inputs, 'r') as fd2:
                inputs = yaml.load(fd2, Loader=yaml.SafeLoader)

            dst_original = inputs['dst']
            inputs['dst'] = f'{folder}/{dst_original}'

            output = {}
            get_slab_E(output, **inputs)
            split_results(output, **inputs)

            name = folder.split('/')[-1]
            result = summarize_results(name, output, **inputs)

            fd1.write(result)


if __name__ == "__main__":
    main()
