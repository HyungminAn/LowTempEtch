from ase.io import read, write
from ase.calculators.mixing import MixedCalculator

import time, sys, os
import numpy as np
import warnings

sys.path.append('/data2/alphalm4/ase_scripts')
from my_functions import get_elements, log_initial_structure, \
         load_calc_gnn, atom_relax, generate_calculator_d3

warnings.filterwarnings(action='ignore')

if __name__=="__main__":
    struct_path = sys.argv[1]
    model = sys.argv[2]
    d3_on = True

    time_init = time.time()
    logfile = open(f'log', 'w', buffering=1)
    extxyz_file = 'opt.extxyz'

    # Initialization
    with open(extxyz_file, 'w') as f:
        pass

    atoms = read(struct_path, format='vasp')
    elements_list = get_elements(atoms)
    elements = ' '.join(elements_list)

    logfile.write(f"{elements}\n")
    log_initial_structure(atoms, logfile)

    logfile.write('######################\n')
    logfile.write('##   Relax starts   ##\n')
    logfile.write('######################\n')

    calc_gnn, model_path = load_calc_gnn(model)
    logfile.write(f'GNN loaded: {model}\n')
    logfile.write(f'GNN path: {model_path}\n')

    if d3_on:
        # assert model[:3] != 'off', "MACE-OFF23 model implicitly include D3 correction"
        if model[:3] == 'off':
            print("MACE-OFF23 model implicitly include D3 correction")
            print("Turn of D3 correction")
            d3_on = False
            calc = calc_gnn

        else:
            calc_d3 = generate_calculator_d3(elements)
            logfile.write(f'D3BJ implemented\n')
            calc = MixedCalculator(calc_gnn, calc_d3, 1, 1)
    else:
        calc = calc_gnn

    atoms = atom_relax(atoms,
                       calc,
                       fmax=0.02,
                       steps=5000,
                       cell_relax=False,
                       fix_sym=False,
                       symprec=None,
                       logfile=logfile,
                       extxyz_file=extxyz_file,
                       )

    write(f'relaxed.POSCAR', atoms)
    logfile.write(f'\nElapsed time: {time.time()-time_init} s\n\n')
    logfile.write('########################\n')
    logfile.write('##  Relax terminated  ##\n')
    logfile.write('########################\n')

    logfile.close()
