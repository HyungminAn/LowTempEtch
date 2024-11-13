import numpy as np
from ase.io import read
import time
import sys
import matplotlib.pyplot as plt
"""
trajectory should be unwrapped

Original code by Jiho Lee
Modified by Hyungmin An, 2024. 06. 03
"""


def msd_aimd(path, step, atomic_number):
    atoms_list = read(path, index='::'+str(step), format='vasp-xdatcar')
    N = len(atoms_list)

    positions = [
        atoms[atoms.get_atomic_numbers() == atomic_number].get_positions()
        for atoms in atoms_list]
    positions = np.array(positions)

    MSD = []
    for delt in range(1, N):
        # Calculate differences using broadcasting
        # resulting in an array with shape (N-delt, N_Li, 3)
        diffs = positions[delt:] - positions[:-delt]

        # Calculate squared distances
        squared_distances = np.sum(diffs**2, axis=-1)
        # Average over particles and time origins
        msd_avg = np.mean(squared_distances)
        # Standard deviation over particles and time origins
        msd_std = np.std(squared_distances)

        MSD.append((delt, msd_avg, msd_std))

        print("delt: ", delt, "msd_avg: ", msd_avg, "msd_std: ", msd_std)

    # Save MSD values
    with open('msd_avg_AIMD', 'w') as f:
        for delt, msd, std in MSD:
            f.write(f"{delt * step} {msd:.5f} {std:.5f}\n")


def msd_md(path, step, atom_type):
    atoms_list = read(path, index='::'+str(step), format='lammps-dump-text')
    N = len(atoms_list)

    positions = [
        atoms[atoms.numbers == atom_type].get_positions()
        for atoms in atoms_list]
    positions = np.array(positions)

    MSD = []
    for delt in range(1, N):
        # Calculate differences using broadcasting,
        # resulting in an array with shape (N-delt, N_Li, 3)
        diffs = positions[delt:] - positions[:-delt]

        # Calculate squared distances
        squared_distances = np.sum(diffs**2, axis=-1)

        # Average over particles and time origins
        msd_avg = np.mean(squared_distances)
        # Standard deviation over particles and time origins
        msd_std = np.std(squared_distances)

        MSD.append((delt, msd_avg, msd_std))

        print("delt: ", delt, "msd_avg: ", msd_avg, "msd_std: ", msd_std)

    # Save MSD values
    with open('msd_avg_MD', 'w') as f:
        for delt, msd, std in MSD:
            f.write(f"{delt * step} {msd:.5f} {std:.5f}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python time_average_unwrap.py method path")
        print("method: AIMD or MD")
        print("path: path to the trajectory file")
        sys.exit()

    method, path = sys.argv[1:3]

    step = 10
    atomic_number = 9
    atom_type = 9

    start_time = time.time()

    if method == 'AIMD':
        msd_aimd(path, step, atomic_number)

    if method == 'MD':
        msd_md(path, step, atom_type)

    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
