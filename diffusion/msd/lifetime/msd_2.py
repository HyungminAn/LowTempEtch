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


@contextmanager
def print_progress(job_name):
    '''
    Simple function to print progress
    '''
    print(f'{job_name}...', end='', flush=True)
    yield
    print('Done', flush=True)


def count_HF_molecules(idx, image, count_dict, **params):
    '''
    Count the number of HF molecules in the image
    and get the indices of the atoms in the HF molecules
    '''
    d_HF = params.get('d_HF')

    pos = image.get_positions()
    cell = image.get_cell()

    idx_H = [atom.index for atom in image if atom.symbol == 'H']
    idx_F = [atom.index for atom in image if atom.symbol == 'F']
    _, D_len = get_distances(pos[idx_H], p2=pos[idx_F], cell=cell, pbc=True)
    n_rows, n_cols = D_len.shape

    for row in range(n_rows):
        for col in range(n_cols):
            if D_len[row, col] > d_HF:
                continue

            key = (idx_H[row], idx_F[col])
            if count_dict.get(key) is None:
                count_dict[key] = [idx]
            count_dict[key].append(idx)


def split_time_interval(v, **params):
    '''
    Split the time interval into sub-intervals
    '''
    intervals = []
    idx_shift = 0
    for idx in range(len(v) - 1):
        if v[idx + 1] - v[idx] > 1:
            intervals.append(v[idx_shift:idx + 1])
            idx_shift = idx + 1
    if not intervals:
        intervals.append(v)

    intervals = [v for v in intervals if len(v) >= params['min_lifetime']]
    return intervals


def load_HF_lifetime(dump, **params):
    '''
    Load data from pickle file or process the data
    '''
    filename = 'lifetime.dat'
    if os.path.exists(filename):
        with print_progress('Loading dat'):
            with open(filename, 'rb') as f:
                dat = pickle.load(f)
        return dat

    with print_progress('Processing dat'):
        dat = {}
        for idx, image in enumerate(dump):
            count_HF_molecules(idx, image, dat, **params)

    with print_progress('Writing dat'):
        with open(filename, 'wb') as f:
            pickle.dump(dat, f)

    return dat


def get_squared_displacement(dump, atom_idx, **params):
    '''
    Get the data from the dump file
    '''
    step_per_image = params.get('step_per_image')
    time_per_step = params.get('time_per_step')
    time_step = time_per_step * step_per_image
    print_step = params.get('print_step')

    dat = []
    pos_before = None
    v = np.zeros(3)

    for idx, image in enumerate(dump):
        pos_now = image.get_positions()[atom_idx]
        pos_before = pos_before if pos_before is not None else pos_now
        v_d, _ = get_distances(
            pos_before, p2=pos_now, cell=image.get_cell(), pbc=True)
        v += v_d[0][0]

        x = idx * time_step
        y = np.square(np.linalg.norm(v[:2]))  # 2D case
        # dat.append((x, y))
        dat.append(y)
        pos_before = pos_now

        if print_step and idx % print_step == 0:
            print(f'idx: {idx}, x: {x}, y: {y}')

    return dat


def get_average(dat, **params):
    '''
    Get the average of the data, whose length is not the same
    '''
    max_length = max(len(d) for d in dat)
    avg = np.zeros(max_length)
    count = np.zeros(max_length)
    for d in dat:
        for i, v in enumerate(d):
            avg[i] += v
            count[i] += 1

    avg = np.array([
        v / n for v, n in zip(avg, count) if n >= params['n_min_sample']])
    return avg


def load_msd(path_dump_list, **params):
    '''
    Load the msd from the pickle file if it exists, otherwise calculate it
    '''
    if os.path.exists('msd'):
        with print_progress('Loading msd'):
            with open('msd', 'rb') as f:
                msd = pickle.load(f)
                return msd

    with print_progress('Calculating msd'):
        dat = []
        for path_dump in path_dump_list:
            dump = read(path_dump, index=':')
            lifetime = load_HF_lifetime(dump, **params)

            for (idx_H, idx_F), time_list in lifetime.items():
                for time_interval in split_time_interval(time_list, **params):
                    start, end = time_interval[0], time_interval[-1]
                    dat.append(get_squared_displacement(
                        dump[start:end], idx_H, **params))

        # Get the average for each component of *dat*
        msd = get_average(dat, **params)

    with print_progress('Saving msd'):
        with open('msd', 'wb') as f:
            pickle.dump(msd, f)

    return msd


def plot_msd(dat, **params):
    step_per_image = params.get('step_per_image')
    time_per_step = params.get('time_per_step')
    time_step = time_per_step * step_per_image

    x = [i * time_step for i in range(len(dat))]
    y = dat

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
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


def main():
    params = {
        'step_per_image': 100,
        'time_per_step': 0.001,  # ps unit
        'print_step': 100,

        'd_HF': 1.5,
        'min_lifetime': 10,
        'n_min_sample': 1,
    }
    if len(sys.argv) < 2:
        print('Usage: python msd.py dump1 dump2 dump3 ...')
        sys.exit(1)

    path_dump_list = sys.argv[1:]
    print(path_dump_list)

    msd = load_msd(path_dump_list, **params)
    plot_msd(msd, **params)


if __name__ == '__main__':
    main()
