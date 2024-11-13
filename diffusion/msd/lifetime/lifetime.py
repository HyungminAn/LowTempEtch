import os
import sys
import functools
import time
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import pickle

from ase.io import read
from ase.geometry import get_distances


def log_function_call(func, indent_level=[0]):
    '''
    Check function runtime and print start/end time
    '''
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

    line = ""
    for interval in intervals:
        if len(interval) < params['min_lifetime']:
            continue

        line += f"[{interval[0]} {interval[-1]} {len(interval)}]"
    return line


@contextmanager
def print_progress(job_name):
    print(f'{job_name}...', end='', flush=True)
    yield
    print('Done', flush=True)


def load_data(dump, **params):
    '''
    Load data from pickle file or process the data
    '''
    if os.path.exists('dat'):
        with print_progress('Loading dat'):
            with open('dat', 'rb') as f:
                dat = pickle.load(f)
    else:
        with print_progress('Processing dat'):
            dat = {}
            for idx, image in enumerate(dump):
                count_HF_molecules(idx, image, dat, **params)

        with print_progress('Writing dat'):
            with open('dat', 'wb') as f:
                pickle.dump(dat, f)


@log_function_call
def plot(dat, **params):
    '''
    plot HF lifetime
    '''
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    conv_factor = params['conv_factor']

    shift = 0
    for k, v in dat.items():
        line = split_time_interval(v, **params)
        if not line:
            continue

        x = np.array(v) * conv_factor
        y = np.arange(0, len(x)) * conv_factor + shift
        ax.scatter(x, y, label=f'{k[0]}-{k[1]}', s=1)
        shift += params['shift']
        print(f'{k} {len(v)} {line}')

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('HF LifeTime (ps)')
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig('lifetime.png')


@log_function_call
def main():
    params = {
        'index': ':',
        'd_HF': 1.2,
        'conv_factor': 0.1,
        'shift': 0.04,
        'min_lifetime': 10,
    }

    path_dump = sys.argv[1]
    dump = read(path_dump, index=params['index'])
    dat = load_data(dump, **params)
    plot(dat, **params)


if __name__ == '__main__':
    main()
