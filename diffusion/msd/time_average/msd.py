import time
import sys

import numpy as np
from ase.io import read
from ase.geometry import get_distances
import cProfile
"""
trajectory should be unwrapped

Original code by Jiho Lee
Modified by Hyungmin An, 2024. 06. 03
"""


def load_data(**inputs):
    paths = inputs['paths']
    method = inputs['method']
    step = inputs['step']
    atom_type = inputs['atom_type']

    if method == 'AIMD':
        atoms_format = 'vasp-out'
    elif method == 'MD':
        atoms_format = 'lammps-dump-text'
    else:
        raise ValueError("method should be AIMD or MD")

    atoms_list = []
    for path in paths:
        atoms = read(path, format=atoms_format, index=':')
        atoms_list.extend(atoms)
    atoms_list = atoms_list[::step]

    idx_F = np.array([
        atom.index for atom in atoms_list[0] if atom.symbol == atom_type])

    mask_F_SiF, mask_F_HF, mask_F_other = separate_F_index(**inputs)
    positions = [
        atoms[idx_F].get_positions()[:, 0:2]
        for atoms in atoms_list]
    positions = np.array(positions)

    return positions, mask_F_SiF, mask_F_HF, mask_F_other


def msd(**inputs):
    MSD = []
    positions, mask_F_SiF, mask_F_HF, mask_F_other = load_data(**inputs)
    step = inputs['step']
    N = len(positions)

    for delt in range(1, N):
        # Calculate differences using broadcasting
        # resulting in an array with shape (N-delt, N_Li, 3)
        diffs = positions[delt:] - positions[:-delt]

        # Calculate squared distances
        squared_distances = np.sum(diffs**2, axis=-1)
        sd_F_SiF = squared_distances[:, mask_F_SiF]
        sd_F_HF = squared_distances[:, mask_F_HF]
        sd_F_other = squared_distances[:, mask_F_other]

        # Average over particles and time origins
        msd_avg = np.mean(squared_distances)
        msd_avg_F_SiF = np.mean(sd_F_SiF)
        msd_avg_F_HF = np.mean(sd_F_HF)
        msd_avg_F_other = np.mean(sd_F_other)

        # Standard deviation over particles and time origins
        msd_std = np.std(squared_distances)
        msd_std_F_SiF = np.std(sd_F_SiF)
        msd_std_F_HF = np.std(sd_F_HF)
        msd_std_F_other = np.std(sd_F_other)

        dat = (
            delt, msd_avg, msd_std,
            msd_avg_F_SiF, msd_std_F_SiF,
            msd_avg_F_HF, msd_std_F_HF,
            msd_avg_F_other, msd_std_F_other,
            )
        MSD.append(dat)

        line = f"delt * step: {delt * step:5d} "
        line += f"msd_avg: {msd_avg:10.5f} msd_std: {msd_std:10.5f} "
        line += f"avg_SiF: {msd_avg_F_SiF:10.5f} "
        line += f"msd_SiF: {msd_std_F_SiF:10.5f} "
        line += f"avg_HF: {msd_avg_F_HF:10.5f} msd_HF: {msd_std_F_HF:10.5f} "
        line += f"avg_other: {msd_avg_F_other:10.5f} "
        line += f"msd_oth: {msd_std_F_other:10.5f}"
        print(line)

    # Save MSD values
    with open('msd_avg.dat', 'w') as f:
        for dat in MSD:
            delt, msd, std, \
                  msd_F_SiF, std_F_SiF, \
                  msd_F_HF, std_F_HF, \
                  msd_F_other, std_F_other = dat
            line = f"{delt * step} {msd:10.5f} {std:10.5f} "
            line += f"{msd_F_SiF:10.5f} {std_F_SiF:10.5f} "
            line += f"{msd_F_HF:10.5f} {std_F_HF:10.5f} "
            line += f"{msd_F_other:10.5f} {std_F_other:10.5f}\n"
            f.write(line)


def separate_F_index(**inputs):
    paths = inputs['paths']
    d_SiF = inputs['d_SiF']
    d_HF = inputs['d_HF']

    path_initial_poscar = paths[0]
    poscar = read(path_initial_poscar)
    pos = poscar.get_positions()
    _, D_len = get_distances(pos, cell=poscar.get_cell(), pbc=True)
    idx_F = [atom.index for atom in poscar if atom.symbol == 'F']
    idx_Si = [atom.index for atom in poscar if atom.symbol == 'Si']
    idx_H = [atom.index for atom in poscar if atom.symbol == 'H']

    idx_F_SiF = [i for i in idx_F if np.min(D_len[i, idx_Si]) < d_SiF]
    idx_F_HF = [i for i in idx_F if np.min(D_len[i, idx_H]) < d_HF]
    idx_F_other = [i for i in idx_F if i not in idx_F_SiF + idx_F_HF]

    mask_F_SiF = np.array([i in idx_F_SiF for i in idx_F])
    mask_F_HF = np.array([i in idx_F_HF for i in idx_F])
    mask_F_other = np.array([i in idx_F_other for i in idx_F])

    return mask_F_SiF, mask_F_HF, mask_F_other


def main():
    if len(sys.argv) < 3:
        print("Usage: python time_average_unwrap.py method paths")
        print("method: AIMD or MD")
        print("path: path to the trajectory file")
        sys.exit()

    method, paths = sys.argv[1], sys.argv[2:]

    inputs = {
        'method': method,
        'paths': paths,
        'step': 10,
        'atom_type': 'F',
        'd_SiF': 2.0,
        'd_HF': 1.1,
    }

    start_time = time.time()

    msd(**inputs)

    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    cProfile.run('main()', 'result.prof')
    # main()
