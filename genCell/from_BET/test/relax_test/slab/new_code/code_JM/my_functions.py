from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.vasp import Vasp
from sevenn.sevennet_calculator import SevenNetCalculator
# from mace.calculators import MACECalculator

from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import ExpCellFilter
# from ase.filters import ExpCellFilter
from ase.io.trajectory import Trajectory
from ase.spacegroup.symmetrize import check_symmetry, FixSymmetry

from ase.build.tools import sort
from ase.units import bar

from ase.io import read, write
import time, sys, os
import numpy as np
from copy import deepcopy


# def generate_calculator(filename, elements):

#     parameters = {
#         'pair_style': 'e3gnn',
#         'pair_coeff': [f'* * {filename} {elements}'],
#         'neigh_modify': 'every 1 delay 0 check yes',
#     }
#     files = [filename]
#     calculator = LAMMPS(parameters=parameters, files=files, keep_alive=True)

#     return calculator

def generate_calculator_d3(elements):
    d3_dir = "/home/alphalm4/bin/d3-CUDA_FP32/lammps_test"
    r0ab_path = os.path.join(d3_dir, "r0ab.csv")
    d3_pars_path = os.path.join(d3_dir, "d3_pars.csv")

    parameters = {
        'units'           : 'metal',
        'box'             : 'tilt large',
        'newton'          : 'on',
        'pair_style'      : 'd3 9000 1600 d3_damp_bj',
        'pair_coeff'      : [
            f'* *     {r0ab_path} {d3_pars_path} pbe {elements}'
        ],
        'neigh_modify'    : 'every 1 delay 0 check yes',
    }

    specorder = elements.split(' ')

    calculator = LAMMPS(
        parameters=parameters,
        # no_data_file=False,
        specorder=specorder,
        keep_alive=False,
        verbose=True,
        write_velocities=True,
        always_triclinic=True,
        binary_dump=False,
        # tmp_dir='temp_lammps',
    )

    return calculator

# def generate_calculator_vasp_debug(com=None, ncore=1):

#     # kpts: gamma
#     # pp: PBE_PAW, use Li_sv for Li
#     vasp_params = {

#         'xc': 'PBE',
#         'kpts': 1,
#         'setups': {
#             'Li': '_sv',
#         },
#         'txt': 'stdout.ase',

#         'system': 'relax_dimer',
#         'nwrite': 2,
#         # 'istart': 0,         # 0: scratch, 1 (default): read WAVECAR
#         'icharg': 1,         # 0: from ini wav, 1: from CHGCAR, 2: from atomic charge
#         'iniwav': 1,

#         'encut': 500,
#         'prec': 'Normal',
#         'addgrid': False,

#         'nelm': 200,
#         'ediff': 1E-06,
#         'lreal': 'Auto',
#         'algo': 'Normal',
#         # 'lwave':  False,      # True (defalt): write WAVECAR
#         # 'lcharg': False,      # True (default): write CHG, CHGCAR

#         # 'nsw': 500,
#         # 'ibrion': 2,
#         # 'ediffg': -0.02,
#         # 'isif': 2,
#         'isym': 0,

#         'ismear': 0,
#         'sigma': 0.05,
#         'lorbit': 11,

#         'ispin': 2,
#         'ivdw': 12,

#     }

#     if com is not None:
#         vasp_params['ldipol'] = True
#         vasp_params['idipol'] = 4
#         vasp_params['dipol'] = com
#         print("Dipole correction is set")

#     if isinstance(ncore, int) and ncore > 1:
#         vasp_params['ncore'] = int(ncore)

#     return Vasp(**vasp_params,
#                 ignore_constraints=True)

# def generate_calculator_vasp(com=None, ncore=1):

#     # kpts: gamma
#     # pp: PBE_PAW, use Li_sv for Li
#     vasp_params = {

#         'xc': 'PBE',
#         'kpts': 1,
#         'setups': {
#             'Li': '_sv',
#         },
#         'txt': 'stdout.ase',

#         'system': 'relax_dimer',
#         'nwrite': 2,
#         'istart': 0,         # 0: scratch, 1 (default): read WAVECAR
#         'icharg': 2,         # 0: from ini wav, 1: from CHGCAR, 2: from atomic charge
#         'iniwav': 1,

#         'encut': 500,
#         'prec': 'Normal',
#         'addgrid': False,

#         'nelm': 200,
#         'ediff': 1E-06,
#         'lreal': 'Auto',
#         'algo': 'Normal',
#         'lwave':  False,      # True (defalt): write WAVECAR
#         'lcharg': False,      # True (default): write CHG, CHGCAR

#         # 'nsw': 500,
#         # 'ibrion': 2,
#         # 'ediffg': -0.02,
#         # 'isif': 2,
#         'isym': 0,

#         'ismear': 0,
#         'sigma': 0.05,
#         'lorbit': 11,

#         'ispin': 2,
#         'ivdw': 12,

#     }

#     if com is not None:
#         vasp_params['ldipol'] = True
#         vasp_params['idipol'] = 4
#         vasp_params['dipol'] = com
#         print("Dipole correction is set")

#     if isinstance(ncore, int) and ncore > 1:
#         vasp_params['ncore'] = int(ncore)

#     return Vasp(**vasp_params,
#                 ignore_constraints=True)


def initialize_info():
    for info in ['PotEng', 'volume', 'stress']:
        with open(info, 'w') as f:
            pass

def initialize_info_vasp():
    for info in ['PotEng', 'volume', 'stress', 'magmom']:
        with open(info, 'w') as f:
            pass

def write_extxyz(extxyz_filename, atoms, step):
    forces = atoms.get_forces()
    cell = atoms.get_cell()
    pbc_list = []
    for i in range(3):
        if atoms.pbc[i] == True:
            pbc_list.append("T")
        else:
            pbc_list.append("F")
    pbc_string = " ".join(pbc_list)
    with open(extxyz_filename, 'a') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"Lattice=\"{cell[0][0]:.6f} {cell[0][1]:.6f} {cell[0][2]:.6f} {cell[1][0]:.6f} {cell[1][1]:.6f} {cell[1][2]:.6f} {cell[2][0]:.6f} {cell[2][1]:.6f} {cell[2][2]:.6f}\" ")
            f.write(f"Properties=species:S:1:pos:R:3:forces:R:3 ")
            f.write(f"pbc={pbc_string}\n")
            for i, atom in enumerate(atoms):
                f.write(f"{atom.symbol}  {atom.position[0]:.6f}  {atom.position[1]:.6f}  {atom.position[2]:.6f}  {forces[i][0]:.6f}  {forces[i][1]:.6f}  {forces[i][2]:.6f}\n")

def save_info(atoms, extxyz_filename, step):
    """
    Save atomic positions to a trajectory file and the stress tensor to another file.
    Voigt order: xx, yy, zz, yz, xz, xy
    Saved order: xx, yy, zz, xy, yz, zx
    """

    PotEng = atoms.get_potential_energy()
    volume  =atoms.get_volume()
    stress = atoms.get_stress() / (-bar)
    press  = np.average(stress[0:3])
    with open('PotEng', 'a') as f:
        f.write(f"{step}  {PotEng:.6f}\n")
    with open('volume', 'a') as f:
        f.write(f"{step}  {volume:.6f}\n")
    with open('stress', 'a') as f:
        f.write(f"{step}  {press:.6f}  {stress[0]:.6f}  {stress[1]:.6f}  {stress[2]:.6f}  {stress[5]:.6f}  {stress[3]:.6f}  {stress[4]:.6f}\n")

    write_extxyz(extxyz_filename, atoms, step)

def save_info_vasp(atoms, extxyz_filename, step):
    """
    Save atomic positions to a trajectory file and the stress tensor to another file.
    Voigt order: xx, yy, zz, yz, xz, xy
    Saved order: xx, yy, zz, xy, yz, zx
    """

    PotEng = atoms.get_potential_energy()
    volume  =atoms.get_volume()
    stress = atoms.get_stress() / (-bar)
    magmom = atoms.get_magnetic_moments()
    press  = np.average(stress[0:3])
    with open('PotEng', 'a') as f:
        f.write(f"{step}  {PotEng:.6f}\n")
    with open('volume', 'a') as f:
        f.write(f"{step}  {volume:.6f}\n")
    with open('stress', 'a') as f:
        f.write(f"{step}  {press:.6f}  {stress[0]:.6f}  {stress[1]:.6f}  {stress[2]:.6f}  {stress[5]:.6f}  {stress[3]:.6f}  {stress[4]:.6f}\n")
    with open('magmom', 'a') as f:
        f.write(f"{step}  {magmom}\n")
    write_extxyz(extxyz_filename, atoms, step)

def atom_relax(atoms, calc, fmax=0.02, steps=5000,
               cell_relax=False, fix_sym=False,  symprec=1e-4,
               logfile='-', extxyz_file='opt.extxyz'):

    initial = deepcopy(atoms)
    initial.calc = calc

    if fix_sym:
        initial.set_constraint(FixSymmetry(initial, symprec=symprec))

    if cell_relax:
        ecf = ExpCellFilter(initial)
        opt = LBFGS(ecf, logfile=logfile)
    else:
        opt = LBFGS(initial, logfile=logfile)

    # Define a function to be called at each optimization step
    initialize_info()
    def custom_step_writer():
        step = opt.nsteps
        save_info(initial, extxyz_file, step)

    opt.attach(custom_step_writer)
    opt.run(fmax=fmax, steps=steps)

    return initial

def sort_atoms(atoms):

    elements_list = []
    index_sorted  = []
    _chem_sym = atoms.get_chemical_symbols()

    for el in atoms.get_chemical_symbols():
        if el not in elements_list:
            elements_list.append(el)

    elements_list.sort()
    for element in elements_list:
        index_sorted += [i for i, x in enumerate(_chem_sym) if x == element]

    return atoms[index_sorted], elements_list

def get_elements(atoms):

    elements_list = []

    for el in atoms.get_chemical_symbols():
        if el not in elements_list:
            elements_list.append(el)

    return elements_list

def log_initial_structure(atoms, logfile='-'):

    _cell = atoms.get_cell()
    logfile.write("Cell\n")
    for ilat in range(0,3):
        logfile.write(f"{_cell[ilat][0]:.6f}  {_cell[ilat][1]:.6f}  {_cell[ilat][2]:.6f}\n")
    _pbc = atoms.get_pbc()
    logfile.write("PBCs\n")
    logfile.write(f"{_pbc[0]}  {_pbc[1]}  {_pbc[2]}\n")
    _pos = atoms.get_positions()
    _mass = atoms.get_masses()
    _chem_sym = atoms.get_chemical_symbols()
    logfile.write("Atoms\n")
    for i in range(len(_chem_sym)):
        logfile.write(f"{_chem_sym[i]}  {_mass[i]:.6f}  {_pos[i][0]:.6f}  {_pos[i][1]:.6f}  {_pos[i][2]:.6f}\n")

def log_symmetry(atoms, spg_prec=1.0e-4, logfile='-'):
    spg_data = check_symmetry(atoms, spg_prec, verbose=True)
    logfile.write("======= Symmetry info =======\n")
    logfile.write(f"\tprec: {spg_prec:.3e}\n")
    logfile.write(f"\tinternational (Hermann-Mauguin): {spg_data['international']}\n")
    logfile.write(f"\tSG#: {spg_data['number']}\n")
    logfile.write(f"============================\n")

def load_calc_gnn(model:str):

    mace_models = dict(
        mp_small   = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_small.model",
        mp_medium  = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_medium.model",
        mp_large   = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_large.model",
        off_small  = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_small.model",
        off_medium = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_medium.model",
        off_large  = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_large.model",

        mp_small_64   = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_small.model",
        mp_medium_64  = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_medium.model",
        mp_large_64   = "/data/tmp/GNN_pretrained/MS/Mace/MACE_MP/mace_large.model",
        off_small_64  = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_small.model",
        off_medium_64 = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_medium.model",
        off_large_64  = "/data/tmp/GNN_pretrained/MS/Mace/MACE-OFF/MACE-OFF23_large.model",
    )

    sevennet_models = dict(
        # sevennet_0      = "/home/alphalm4/bin/SevenNet_correct_240416/SevenNet_0/checkpoint_sevennet_0.pth", # sevennet_m3
        # sevennet_0_old  = "/data2/2022_Li_team/liquid_electrolyte/SevenNet_old_checkpoints/7net_0/checkpoint_sevennet_0.pth",
        # sevennet_01     = "/data2/shared_data/pretrained/sevennet0_on_chgnetdb/checkpoint.pth", # sevennet_ch_c
        # sevennet_01_old = "/data2/2022_Li_team/liquid_electrolyte/SevenNet_old_checkpoints/on_chgnet_dataset_0401/sevennet_chgnetD_best.pth",


        sevennet_chg   = "/data2/shared_data/pretrained/7net_chg/checkpoint_best.pth",
        sevennet_m3g   = "/data2/shared_data/pretrained/7net_m3g/checkpoint_best.pth",
        sevennet_m3g_c55 = "/data2/shared_data/pretrained/7net_m3g_c55/checkpoint_best.pth",

        # experimental
        sevennet_m3g_rand1 = "/data2/shared_data/pretrained_experimental/7net_m3g_rand1/checkpoint_best.pth",
        sevennet_m3g_rand2 = "/data2/shared_data/pretrained_experimental/7net_m3g_rand2/checkpoint_best.pth",
        sevennet_m3g_rand3 = "/data2/shared_data/pretrained_experimental/7net_m3g_rand3/checkpoint_best.pth",
        sevennet_m3g_rand4 = "/data2/shared_data/pretrained_experimental/7net_m3g_rand4/checkpoint_best.pth",

        sevennet_m3g_n = "/data2/shared_data/pretrained_experimental/7net_m3g_n/checkpoint_best.pth",

        sevennet_m3g_l3i3 = "/data2/shared_data/pretrained_experimental/7net_m3g_l3i3/checkpoint_best.pth",
        sevennet_m3g_l3i3_rand1 = "/data2/shared_data/pretrained_experimental/7net_m3g_l3i3_rand1/checkpoint_best.pth",
        sevennet_m3g_l3i3_rand2 = "/data2/shared_data/pretrained_experimental/7net_m3g_l3i3_rand2/checkpoint_best.pth",

        sevennet_m3g_l3i3_n = "/data2/shared_data/pretrained_experimental/7net_m3g_l3i3_n/checkpoint_best.pth",

        # sevennet_chgTot = "/data/tmp/chgTot/checkpoint_600.pth",
        sevennet_chgTot = "/data2/alphalm4/chgTot/checkpoint_600.pth",
        # Deprecated
        # XPLOR cutoff bug
        # sevennet_m3_old  = "/data2/2022_Li_team/liquid_electrolyte/SevenNet_old_checkpoints/7net_0/checkpoint_sevennet_0.pth", # sevennet_0_old
        # sevennet_ch_c_old = "/data2/2022_Li_team/liquid_electrolyte/SevenNet_old_checkpoints/on_chgnet_dataset_0401/sevennet_chgnetD_best.pth", # sevennet_01_old
        # sevennet_ch_c = "/data2/shared_data/pretrained/7net_ch_c/checkpoint_best.pth",
    )

    if model in mace_models.keys():
        model_path = mace_models[model]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

#         if model[-3:] == '_64':
#             calc_gnn = MACECalculator(model_paths=model_path,
#                                       dispersion=False,
#                                       default_dtype="float64",
#                                       device='cuda',
#                                       )
#
#         else:
#             calc_gnn = MACECalculator(model_paths=model_path,
#                                     dispersion=False,
#                                     default_dtype="float32",
#                                     device='cuda',
#                                     )

    elif model in sevennet_models.keys():
        model_path = sevennet_models[model]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        calc_gnn = SevenNetCalculator(model=model_path)


    else:
        raise ValueError(f"Model {model} not found")

    print(f"GNN model {model} loaded: {model_path}")
    return calc_gnn, model_path
