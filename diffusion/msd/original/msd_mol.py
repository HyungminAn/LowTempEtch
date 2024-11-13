import os
import sys
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import pickle

from ase.io import read
from ase.geometry import get_distances
"""
Calculates the mean squared displacement of a molecule in a trajectory
Usage : python -.py dump1 dump2 dump3 ...
"""


def get_squared_displacement(path_dump, idx_F, **params):
    '''
    Get the data from the dump file
    '''
    step_per_image = params.get('step_per_image')
    time_per_step = params.get('time_per_step')
    print_step = params.get('print_step')
    idx_to_select = params['idx_to_select']

    dump = read(path_dump, index=idx_to_select)
    dat = []

    pos_F_before = None
    v = np.zeros(3)
    for idx, image in enumerate(dump):
        pos = image.get_positions()
        pos_F = pos[idx_F]
        if pos_F_before is None:
            pos_F_before = pos_F
        cell = image.get_cell()
        v_d, _ = get_distances(pos_F_before, p2=pos_F, cell=cell, pbc=True)
        v_d = v_d[0][0]
        v += v_d

        x = idx * step_per_image * time_per_step
        y = np.square(np.linalg.norm(v[:2]))

        dat.append((x, y))

        pos_F_before = pos_F

        if idx % print_step == 0:
            print(f'idx: {idx}, x: {x}, y: {y}')

    return dat


def plot(dat):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    x, y = zip(*dat)
    ax.plot(x, y)

    # Get the trend line, y = mx (least square fit)
    # Using np.linalg.lstsq
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax.plot(x, m * np.array(x) + c, 'grey', label=f'Fit: {m:.2f}x + {c:.2f}')

    dim = 2  # 2D
    A2_to_cm2 = 1E-16
    ps_to_s = 1E-12
    Diff_coeff = m * A2_to_cm2 / ps_to_s / (2 * dim)  # cm^2/s

    ax.set_title(f'$D = {Diff_coeff:.3e} cm^2/s$')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD $(A^2)$')
    fig.tight_layout()
    fig.savefig('msd.png')


@contextmanager
def print_progress(job_name):
    print(f'{job_name}...', end='', flush=True)
    yield
    print('Done', flush=True)


def load_msd(path_dump_list, **params):
    '''
    Load the msd from the pickle file if it exists, otherwise calculate it
    '''
    idx_H_list = params['idx_H_list']

    if os.path.exists('msd'):
        with print_progress('Loading msd'):
            with open('msd', 'rb') as f:
                msd = pickle.load(f)
                return msd

    with print_progress('Calculating msd'):
        dat_list = []
        for path_dump in path_dump_list:
            for idx_H in idx_H_list:
                dat = get_squared_displacement(path_dump, idx_H, **params)
                dat_list.append(np.array(dat))

        # Get the average of the arrays in dat_list
        msd = np.mean(dat_list, axis=0)

    with print_progress('Saving msd'):
        with open('msd', 'wb') as f:
            pickle.dump(msd, f)

    return msd


def main():
    if len(sys.argv) < 2:
        print('Usage: python msd_mol.py dump1 dump2 dump3 ...')
        sys.exit(1)

    params = {
        'idx_to_select': ':',
        'step_per_image': 100,
        'time_per_step': 0.001,  # ps unit
        'print_step': 100,
        'idx_H_list': [66],
    }
    path_dump_list = sys.argv[1:]
    print(path_dump_list)

    dat = load_msd(path_dump_list, **params)
    plot(dat)


if __name__ == '__main__':
    main()
