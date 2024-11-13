from ase.io import read
from ase.data import atomic_masses, atomic_numbers

def get_mass_info(path_poscar):
    atoms = read(path_poscar)
    symbols = []
    for atom in atoms:
        if atom.symbol not in symbols:
            symbols.append(atom.symbol)

    masses = [atomic_masses[atomic_numbers[symbol]] for symbol in symbols]
    return symbols, masses


def get_reflect_z(path_poscar):
    atoms = read(path_poscar)
    return atoms.get_cell()[2, 2] - 5.0


def write(symbols, masses, reflect_z):
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
variable    fix_h       equal   4.0
region      rFixed      block   INF INF INF INF 0.0 ${fix_h}
group       gBottom     region  rFixed
velocity    gBottom     set     0.0 0.0 0.0
fix         frz_bot     gBottom     setforce    0.0 0.0 0.0

#########################################################
#                 NVT MD variables                      #
#########################################################
variable        T_nvt           equal   250
variable        timestep_nvt    equal   0.001       # ps unit
timestep        ${timestep_nvt}                     # ps unit
variable        time_nvt        equal   10
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
''')
    line += ('''
fix         my_NVT      gMove    nvt     temp    ${T_nvt} ${T_nvt} 0.1

#########################################################
#                       LOG                             #
#########################################################
variable        log_step        equal   100
thermo_style    custom          step pe ke etotal fmax press cpu tpcpu spcpu
thermo          ${log_step}

dump            my_dump     all custom ${log_step} dump.lammps &
                            id type element xu yu zu fx fy fz
dump_modify     my_dump     sort id     element ${element_list}

#########################################################
#                      Run                              #
#########################################################
run ${step_nvt}

write_data      FINAL.coo
''')

    with open('lammps.in', 'w') as f:
        f.write(line)


def main():
    path_poscar = 'POSCAR_merged'
    symbols, masses = get_mass_info(path_poscar)
    reflect_z = get_reflect_z(path_poscar)
    write(symbols, masses, reflect_z)


if __name__ == '__main__':
    main()
