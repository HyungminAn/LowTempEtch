from pathlib import Path
import functools
import time
import subprocess
import shutil
import os
import sys

import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
import yaml

# To import other scripts
script_directory = Path(__file__).parent.absolute()
sys.path.append(str(script_directory))

from write_lmp_input import write_lmp_input_insertion  # noqa: E402
from write_lmp_input import get_element_order  # noqa: E402
from relax_sevenn_d3 import relax  # noqa: E402
from check_disso import is_etchant_dissociated  # noqa: E402
from check_surf_reconst import is_surface_reconstructed  # noqa: E402
from plot import plot  # noqa: E402
from perturb import set_perturbation  # noqa: E402


"""
This code calculates the adsorption energy of the given etchant molecule.

Prerequisite:
    1) relaxed molecule structure
    2) relaxed slab structure

Pipeline consists of:
    1) Make a relaxed structure with additive:
        1-1) insert molecule at random position on slab
        1-2) relaxation of the structure
        1-3) select the slab structure with the lowest energy

    2) Relax the structure with a etchant molecule:
        2-1) insert molecule at random position on slab
        2-2) relaxation of the structure
        2-3) Check for the dissociation of the etchant molecule
             The number of dissociation --> *chemisorption_ratio*
        2-4) Check for surface reconstruction
        2-5) For reconstructed structures, Remove the inserted etchant molecule
        2-6) Re-relax for the new slab structure
        2-7) Re-check for the slab reconstruction

    3) Gather the results and analyze the data:
        3-1) Get the chemisorption ratio
        3-2) Get the physisorption ratio
             = total - chemisorption ratio - reconstruction ratio
        3-3) Get the statistics value for adsorption energy
            : count, average, max, min, stddev, ...
"""


def log_function_call(func, indent_level=[0]):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tab = indent_level[0] * "    "
        indent_level[0] += 1
        clock = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"{tab}Function {func.__name__} started at {clock}")

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        clock = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"{tab}Function {func.__name__} ended at {clock}")
        print(f"{tab}    Elapsed time: {end_time - start_time} seconds")

        indent_level[0] -= 1
        return result
    return wrapper


def pos2lmp(path_poscar, path_dst, elem_order):
    '''
    Read VASP POSCAR and convert to lammps data format
    '''
    poscar = read(path_poscar)
    write(path_dst, poscar, format='lammps-data', specorder=elem_order)


def lmp2pos(path_lmpdat, path_dst, fix_bottom_height=None, elem_order=None):
    '''
    Read lammps data file and convert to VASP POSCAR format
    '''
    lmpdat = read(path_lmpdat, format='lammps-data', atom_style='atomic')
    if fix_bottom_height:
        c = FixAtoms(indices=[
            atom.index for atom in lmpdat
            if atom.position[2] <= fix_bottom_height]
        )
        lmpdat.set_constraint(c)
    write(path_dst, lmpdat, format='vasp', sort=True)


@log_function_call
def run_insertion_LAMMPS(key, seed_max=1000000, **inputs):
    '''
    Repeat
        - Make directory
        - write lammps.in
        - write input.data
        - run lammps
    '''
    dst = inputs["dst"]
    path_slab = inputs[key]["paths"]["slab"]
    path_mol = inputs[key]["paths"]["mol"]

    elem_order = get_element_order(path_mol, path_slab)

    n_insert = inputs[key]["mol_info"]["n_insert"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    path_dst = inputs[key]["paths"]["dst_1"]
    path_lmp = inputs["relax"]["path_lmp_bin"]
    mol_name = inputs[key]["mol_info"]["name"]
    fix_height = inputs["constraint"]["fix_bottom_height"]
    run_short_MD = inputs[key]["mol_info"]["run_short_MD"]
    md_time = inputs[key]["mol_info"]["md_time"]
    md_temp = inputs[key]["mol_info"]["md_temp"]

    if key == "etchant":
        insert_global = True
    else:
        insert_global = False

    for i in range(n_repeat):
        dst = f'{path_dst}/{i}'
        p = Path(dst)
        if p.is_dir():
            continue
        p.mkdir(parents=True, exist_ok=True)

        write_lmp_input_insertion(
            dst, path_mol, path_slab, mol_name, fix_height, n_insert,
            run_short_MD=run_short_MD, md_time=md_time, md_temp=md_temp,
            insert_global=insert_global)

        dst_structure_input = f'{dst}/input.data'
        pos2lmp(path_slab, dst_structure_input, elem_order)

        seeds = np.random.randint(seed_max)
        cmd = f'{path_lmp} -in lammps.in -var SEEDS {seeds} > lammps_{i}.out'
        subprocess.run(cmd, cwd=p, shell=True)


@log_function_call
def run_relaxation_ASE(key, **inputs):
    '''
    Repeat
        - Make directory
        - copy input.yaml
        - run ASE_relax
    '''
    path_mol = inputs[key]["paths"]["mol"]
    path_slab = inputs[key]["paths"]["slab"]
    elem_order = get_element_order(path_mol, path_slab)

    n_repeat = inputs[key]["mol_info"]["n_repeat"]

    dst = inputs["dst"]
    path_src = inputs[key]["paths"]["dst_1"]
    path_dst = inputs[key]["paths"]["dst_2"]
    fix_bottom_height = inputs["constraint"]["fix_bottom_height"]

    perturb_flag = inputs[key]["perturb"]["flag"]

    for i in range(n_repeat):
        dst = f'{path_dst}/{i}'
        dst_poscar = f'{dst}/POSCAR'

        p = Path(dst)
        poscar_relaxed = p / inputs["relax"]["path_relaxed"]
        if poscar_relaxed.exists():
            continue
        p.mkdir(parents=True, exist_ok=True)

        src = f'{path_src}/{i}'
        src_lmp_dat = f'{src}/FINAL.coo'
        lmp2pos(src_lmp_dat, dst_poscar,
                fix_bottom_height=fix_bottom_height,
                elem_order=elem_order)

        is_relax_success = relax(dst_poscar, dst, **inputs)

        if perturb_flag:
            rlx_traj = f'{dst}/' + inputs["relax"]["path_extxyz"]
            scale = inputs[key]["perturb"]["scale"]
            cutoff = inputs[key]["perturb"]["cutoff"]
            poscar_perturb = set_perturbation(
                rlx_traj, path_slab, scale, cutoff, fix_bottom_height)
            os.makedirs(f'{dst}/initial', exist_ok=True)
            for file in os.listdir(dst):
                shutil.move(f'{dst}/{file}', f'{dst}/initial')
            write(dst_poscar, poscar_perturb, format='vasp')
            is_relax_success = relax(dst_poscar, dst, **inputs)


@log_function_call
def run_relaxation_ASE_reconst(key, idx_reconst_dict, **inputs):
    '''
    run_relaxation_ASE function for reconstructed slabs
    '''
    path_dst = inputs[key]["paths"]["dst_3"]
    for i in idx_reconst_dict.keys():
        dst = f'{path_dst}/{i}'
        dst_poscar = f'{dst}/POSCAR'

        p = Path(dst)
        poscar_relaxed = p / inputs["relax"]["path_relaxed"]
        if poscar_relaxed.exists():
            continue

        is_relax_success = relax(dst_poscar, dst, **inputs)


@log_function_call
def select_slab_with_minimal_energy(key, output, **inputs):
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
            i_save = i
            energy_save = energy

    output["E_slab"] = energy_save

    path_dst = inputs[key]["paths"]["dst_3"]
    p_src = Path(f'{path_src}/{i_save}/POSCAR_relaxed')
    p_dst = Path(path_dst)

    shutil.copy(p_src, p_dst)


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
def remove_etchant_molecule(key, idx_reconst_dict, **inputs):
    '''
    Remove etchant molecules and write new POSCAR to calculate
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    fix_height = inputs["constraint"]["fix_bottom_height"]
    path_dst = inputs[key]["paths"]["dst_3"]

    for i, idx_etchant in idx_reconst_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        poscar = read(src, index=-1)

        poscar_copy = poscar.copy()
        while poscar_copy:
            poscar_copy.pop()
        for idx, atom in enumerate(poscar):
            if idx in idx_etchant:
                continue
            poscar_copy.append(atom)

        poscar = poscar_copy
        c = FixAtoms([
            atom.index for atom in poscar if atom.position[2] <= fix_height
        ])
        poscar.set_constraint(c)
        dst = f'{path_dst}/{i}'
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)
        write(f'{dst}/POSCAR', poscar, format='vasp')


@log_function_call
def make_slab_with_additive(output, **inputs):
    '''
    1) Make a relaxed structure with additive:
        1-1) insert molecule at random position on slab
        1-2) relaxation of the structure
        1-3) select the slab structure with the lowest energy
    '''
    key = "additive"
    inputs[key]["paths"]["dst_1"] =\
        inputs[key]["paths"]["dst_1"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_2"] =\
        inputs[key]["paths"]["dst_2"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_3"] =\
        inputs[key]["paths"]["dst_3"] % {'dst': inputs['dst']}

    run_insertion_LAMMPS(key, **inputs)
    run_relaxation_ASE(key, **inputs)
    select_slab_with_minimal_energy(key, output, **inputs)


@log_function_call
def repeat_adsorption(output, **inputs):
    '''
    2) Relax the structure with a etchant molecule:
        2-1) insert molecule at random position on slab
        2-2) relaxation of the structure
        2-3) Check for the dissociation of the etchant molecule
             The number of dissociation --> *chemisorption_ratio*
        2-4) Check for surface reconstruction
        2-5) For reconstructed structures, Remove the inserted etchant molecule
        2-6) Re-relax for the new slab structure
        2-7) Re-check for the slab reconstruction
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

    run_insertion_LAMMPS(key, **inputs)
    run_relaxation_ASE(key, **inputs)

    idx_etchant_dict = check_etchant_dissociation(key, **inputs)
    idx_reconst_dict = check_reconstruction(key, idx_etchant_dict, **inputs)
    remove_etchant_molecule(key, idx_reconst_dict, **inputs)
    run_relaxation_ASE_reconst(key, idx_reconst_dict, **inputs)

    exclude_dict = check_reconstruction_reverse(
            key, idx_reconst_dict, **inputs)
    phys_dict = {**idx_etchant_dict, **idx_reconst_dict}

    output["exclude_dict"] = exclude_dict
    output["phys_dict"] = phys_dict


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
    max_step = inputs["relax"]["max_steps"]

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

    # Get energy of mol
    path_mol = inputs[key]["paths"]["mol"]
    path_mol_dat = '/'.join(path_mol.split('/')[:-1]) + '/log.lammps'
    with open(path_mol_dat, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'Energy initial' in line:
                E_mol = float(lines[idx+1].split()[-1])
                break

    E_slab = output["E_slab"]
    E_ref = E_mol + E_slab

    E_ads_dict = {k: E - E_ref for k, E in E_slabMol_dict.items()}
    plot(E_ads_dict, **inputs)


@log_function_call
def main():
    path_inputs = "./input.yaml"
    with open(path_inputs, 'r') as f:
        inputs = yaml.load(f, Loader=yaml.SafeLoader)

    output = {}
    make_slab_with_additive(output, **inputs)
    repeat_adsorption(output, **inputs)
    summarize_results(output, **inputs)


if __name__ == "__main__":
    main()
