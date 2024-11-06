import os
from pathlib import Path

from utils.log import log_function_call  # noqa: E402

from adsorption.plot.energy import plot  # noqa: E402


def get_slabMol_energy(
        phys_dict, src_no_reconst, src_reconst, dir_reconst, max_step):
    # Get energy of slab + mol
    E_slabMol_dict = {}
    for i in phys_dict.keys():
        if i in dir_reconst:
            dst = f'{src_reconst}/{i}/thermo.dat'
        else:
            dst = f'{src_no_reconst}/{i}/thermo.dat'

        with open(dst, 'r') as f:
            step, E_slabMol, *_ = f.readlines()[-1].split()
            step = int(step)
            E_slabMol = float(E_slabMol)
            if step >= max_step:
                continue
            E_slabMol_dict[i] = E_slabMol

    return E_slabMol_dict


def get_Mol_energy(path_mol):
    # Get energy of mol
    path_mol = Path(path_mol)
    with open(path_mol.parent / 'thermo.dat', 'r') as f:
        step, E_mol, *_ = f.readlines()[-1].split()
        step = int(step)
        E_mol = float(E_mol)

    return E_mol


@log_function_call
def summarize_results(output, **inputs):
    '''
    3) Gather the results and analyze the data:
        3-1) Get the chemisorption ratio
        3-2) Get the physisorption ratio
             = total - chemisorption ratio - reconstruction ratio
        3-4) Get the statistics value for adsorption energy
            : count, average, max, min, stddev, ...
    '''
    exclude_dict = output["exclude_dict"]
    phys_dict = output["phys_dict"]

    key = "etchant"
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    n_total = n_repeat - len(exclude_dict)

    n_chem = n_total - len(phys_dict)
    r_chem = n_chem / n_total
    print(f"Chemisorption ratio: {r_chem}")

    src_no_reconst = inputs[key]["paths"]["dst_2"]
    src_reconst = inputs[key]["paths"]["dst_3"]
    dir_reconst = os.listdir(src_reconst) if os.path.isdir(src_reconst) else []
    max_step = inputs["relax"]["options"]["max_steps"]

    E_slabMol_dict = get_slabMol_energy(
        phys_dict, src_no_reconst, src_reconst, dir_reconst, max_step)

    path_mol = inputs[key]["paths"]["mol"]
    E_mol = get_Mol_energy(path_mol)

    E_slab = output["E_slab"]
    E_ref = E_mol + E_slab

    E_ads_dict = {k: E - E_ref for k, E in E_slabMol_dict.items()}
    plot(E_ads_dict, **inputs)
