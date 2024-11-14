import os
import subprocess

import numpy as np

from ase.io import read, write
from ase.data import atomic_masses, atomic_numbers
from ase.constraints import FixAtoms


class MolecularDynamicsSimulator():
    def __init__(self, inputs):
        self.dst = inputs['dst']
        self.path_poscar = inputs['path_poscar']
        self.path_lmp = inputs['path_lmp']
        self.path_potential = inputs['path_potential']

        self.md_time = inputs['md_time']
        self.md_temp = inputs['md_temp']
        self.fix_h = inputs['fix_h']

    def run(self):
        if os.path.exists(f'{self.dst}/FINAL.coo'):
            return

        self._write_lammps_input()
        _pos2lmp(self.path_poscar, self.dst)

        run_options = {
            'check': True,
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
            'shell': True,
            'cwd': self.dst,
        }
        cmd = f"{self.path_lmp} -in lammps.in"
        cmd += f" -var path_potential {self.path_potential}"
        cmd += f" -var SEEDS {np.random.randint(1, 1000000)}"
        print(cmd)
        subprocess.run(cmd, **run_options)

        _lmp2pos(f'{self.dst}/FINAL.coo', self.dst, self.fix_h)

    def _write_lammps_input(self):
        symbols, masses = _get_mass_info(self.path_poscar)
        reflect_z = _get_reflect_z(self.path_poscar)

        line = ('''
#########################################################
#                      Basic Input                      #
#########################################################
units           metal     # K, bar, ps, A
box             tilt large
boundary        p p f
newton          on
read_data       input.data
''')
        for idx, mass in enumerate(masses):
            line += f'mass        {idx+1}   {mass}\n'
        symbols_str = ' '.join(symbols)

        line += f'''
variable        element_list    string  "{symbols_str}"
'''
        line += ('''
#########################################################
#                SEVENNET-0 + D3  hybrid                #
#########################################################
pair_style      hybrid/overlay      e3gnn    d3 9000 1600 damp_bj pbe
pair_coeff      * * e3gnn ${path_potential}  ${element_list}
pair_coeff      * * d3 ${element_list}

#########################################################
#                   Slab settings                       #
#########################################################
''')
        line += (f'''
variable    fix_h       equal   {self.fix_h}
''')
        line += ('''
region      rFixed      block   INF INF INF INF 0.0 ${fix_h}
group       gBottom     region  rFixed
velocity    gBottom     set     0.0 0.0 0.0
fix         frz_bot     gBottom     setforce    0.0 0.0 0.0

#########################################################
#                 NVT MD variables                      #
#########################################################
''')
        line += (f'''
variable        T_nvt           equal   {self.md_temp}
variable        time_nvt        equal   {self.md_time}
''')
        line += ('''
variable        timestep_nvt    equal   0.001       # ps unit
timestep        ${timestep_nvt}                     # ps unit
variable        step_nvt        equal   $(round(v_time_nvt/v_timestep_nvt))

#########################################################
#                 NVE + NVT MD                          #
#########################################################
region      rMove      block   EDGE EDGE   EDGE EDGE   ${fix_h} EDGE
group       gMove      region  rMove
velocity    gMove      create ${T_nvt} ${SEEDS} dist gaussian

''')
        line += (f'''
variable    h_reflect   equal   {reflect_z}
''')
        line += ('''
fix         top         all wall/reflect zhi ${h_reflect}
fix         my_NVT      gMove    nvt     temp    ${T_nvt} ${T_nvt} 0.1

#########################################################
#                       LOG                             #
#########################################################
variable        log_step        equal   100
thermo_style    custom          step pe ke etotal fmax press cpu tpcpu spcpu
thermo          ${log_step}
variable        path_dump      string  dump.lammps
variable        path_output    string  FINAL.coo
dump            my_dump     all custom ${log_step} ${path_dump} &
                            id type element xu yu zu fx fy fz
dump_modify     my_dump     sort id     element ${element_list}

#########################################################
#                      Run                              #
#########################################################
run ${step_nvt}

write_data      ${path_output}
''')

        with open(f'{self.dst}/lammps.in', 'w') as f:
            f.write(line)


def _pos2lmp(path_poscar, dst):
    atoms = read(path_poscar)
    write(f'{dst}/input.data', atoms, format='lammps-data')


def _lmp2pos(path_lmpdat, dst, fix_h):
    atoms = read(path_lmpdat, format='lammps-data')
    c = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] < fix_h])
    atoms.set_constraint(c)
    write(f'{dst}/POSCAR_after_md', atoms, format='vasp')


def _get_mass_info(path_poscar):
    atoms = read(path_poscar)
    symbols = []
    for atom in atoms:
        if atom.symbol not in symbols:
            symbols.append(atom.symbol)

    masses = [atomic_masses[atomic_numbers[symbol]] for symbol in symbols]
    return symbols, masses


def _get_reflect_z(path_poscar):
    atoms = read(path_poscar)
    return atoms.get_cell()[2, 2] - 5.0
