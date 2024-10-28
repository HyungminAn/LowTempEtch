import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ase.io import read


def get_element_order(path_mol, path_slab):
    '''
    Get the order of chemical symbols including
    '''
    elem_order = []

    slab = read(path_slab)
    [elem_order.append(i) for i in slab.get_chemical_symbols()
     if i not in elem_order]

    mol = read(path_mol, format='lammps-dump-text', index='-1')
    [elem_order.append(i) for i in mol.get_chemical_symbols()
     if i not in elem_order]

    return elem_order


def write_mol(mol_name, path_mol_dump, elem_order, path_dst):
    '''
    Write molecule information for the molecule insertion in LAMMPS
    '''
    poscar = read(path_mol_dump, index=-1)
    elem_order_dict = {elem: i+1 for i, elem in enumerate(elem_order)}
    elements_expanded = poscar.get_chemical_symbols()
    type_list = [elem_order_dict[i] for i in elements_expanded]

    n_atoms = len(poscar)
    pos = poscar.get_positions()
    with open(f'{path_dst}/mol_{mol_name}', 'w') as f:
        w = f.write

        w(f"# {mol_name}\n\n")
        w(f"{n_atoms} atoms\n\n")

        w("Coords\n\n")
        for idx, xyz in enumerate(pos):
            w(f"{idx+1} {xyz[0]} {xyz[1]} {xyz[2]}\n")
        w("\n")

        w("Types\n\n")
        for idx, elem_type in enumerate(type_list):
            w(f"{idx+1} {elem_type}\n")
        w("\n")

        w("Masses\n\n")
        for idx, elem in enumerate(elements_expanded):
            mass = atomic_masses[atomic_numbers[elem]]
            w(f"{idx+1} {mass}\n")


def write_lmp_input_insertion(
        path_dst, path_mol, path_slab, mol_name, fix_height, n_insert,
        run_short_MD=False, md_time=None, md_temp=None, insert_global=False):
    '''
    Write input script for LAMMPS calculation,
    and also includes writing molecule informtaion.
    '''
    elem_order = get_element_order(path_mol, path_slab)
    write_mol(mol_name, path_mol, elem_order, path_dst)

    lines = '''
#LAMMPS
#########################################################
#                      Basic Input                      #
#########################################################
units           metal     # K, bar, ps, A
box             tilt large
boundary        p p f
newton          on

read_data       input.data
'''

    for elem in elem_order:
        mass = atomic_masses[atomic_numbers[elem]]
        lines += f'variable m_{elem} equal {mass}\n'
    element_list = ' '.join(elem_order)
    lines += f'\nvariable element_list string "{element_list}"\n\n'
    for idx, elem in enumerate(elem_order):
        lines += f'mass  {idx+1}  ${{m_{elem}}}\n'


    if run_short_MD:
        path_r0ab = \
            "/data2/andynn/lammps_d3/13_rewrite_csv/r0ab_new.csv"
        path_c6ab = \
            "/data2/andynn/lammps_d3/13_rewrite_csv/d3_pars.csv"

        lines += f'''
#########################################################
#                     Pair potential                    #
#########################################################
variable        path_r0ab       string   "{path_r0ab}"
variable        path_c6ab       string   "{path_c6ab}"
variable        cutoff_d3       equal   9000
variable        cutoff_d3_CN    equal   1600
variable        func_type       string  "pbe"
variable        damping_type    string  "d3_damp_bj"
variable        path_potential  getenv  SEVENNET_0
pair_style      hybrid/overlay &
                e3gnn &
                d3   ${{cutoff_d3}}  ${{cutoff_d3_CN}}  ${{damping_type}}
pair_coeff      * *  e3gnn  ${{path_potential}}  ${{element_list}}
pair_coeff      * *  d3 ${{path_r0ab}} ${{path_c6ab}}&
                        ${{func_type}} ${{element_list}}

#########################################################
#                   Slab settings                       #
#########################################################
variable    fix_h       equal   {fix_height}
region      rFixed      block   INF INF INF INF 0.0 ${{fix_h}}
group       gBottom     region  rFixed
velocity    gBottom     set     0.0 0.0 0.0
fix         frz_bot     gBottom     setforce    0.0 0.0 0.0
'''

    if insert_global:
        lines += f'''
#########################################################
#                   Insert molecules                    #
#########################################################
molecule    my_mol      "mol_{mol_name}"
region      rDepo       block   EDGE EDGE  EDGE EDGE  EDGE EDGE
fix         fDepo       all     deposit   {n_insert} 0 1 ${{SEEDS}} &
                        region  rDepo    mol my_mol &
                        global  2.0 2.0
'''
    else:
        lines += f'''
#########################################################
#                   Insert molecules                    #
#########################################################
molecule    my_mol      "mol_{mol_name}"
region      rDepo       block   EDGE EDGE  EDGE EDGE  EDGE EDGE
fix         fDepo       all     deposit   {n_insert} 0 1 ${{SEEDS}} &
                        region  rDepo    mol my_mol &
                        local  2.0 3.0 2.0
'''

    if run_short_MD:
        lines += f'''
#########################################################
#                 NVT MD variables                      #
#########################################################
variable        T_nvt           equal   {md_temp}
variable        timestep_nvt    equal   0.001       # ps unit
timestep        ${{timestep_nvt}}                   # ps unit
variable        time_nvt        equal   {md_time}
variable        step_nvt        equal   $(round(v_time_nvt/v_timestep_nvt))

variable        log_step        equal   100
thermo          ${{log_step}}
thermo_style    custom step temp pe ke etotal press vol
dump            my_dump all custom ${{log_step}} dump.lammps id type x y z vx vy vz

#########################################################
#                 NVT MD                                #
#########################################################
# Required                                              #
# 1) Set the Move region & group, to set NVT MD         #
# 2) If reflection required, check it.                  #
#########################################################
region      rMove       block   EDGE EDGE   EDGE EDGE   ${{fix_h}} EDGE
group       gMove       region  rMove
velocity    gMove       create ${{T_nvt}} ${{SEEDS}} dist gaussian

variable    h_reflect   equal    $(zhi)-5.0
fix         top         all wall/reflect zhi ${{h_reflect}}

variable    nvt_tdamp   equal     $(100*v_timestep_nvt)
fix         my_NVT      gMove    nvt     temp    ${{T_nvt}} ${{T_nvt}} ${{nvt_tdamp}}

run         ${{step_nvt}}
'''
    else:
        lines += f'''run         {n_insert}'''

    lines += '''
#########################################################
#                     Finalize                          #
#########################################################
write_data      FINAL.coo
'''

    with open(f'{path_dst}/lammps.in', 'w') as f:
        w = f.write
        w(lines)
