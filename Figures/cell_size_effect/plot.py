import matplotlib.pyplot as plt
import numpy as np


def read_data(path_log):
    with open(path_log, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if 'Loop time' in line:
            nions = int(line.split()[-2])
            continue

        if 'PotEng' in line:
            lines = lines[idx+1:]
            lines = lines[:101]
            break
    result = []
    for line in lines:
        energy = float(line.split()[1])
        result.append(energy / nions)
    result = np.array(result)

    return result


def plot(data_original, data_large):
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    x = np.arange(len(data_original)) * 0.1
    ax.plot(x, data_original, label='small cell\n' + r'($10 \times 10 \times 40 \AA^3 $, 186 atoms)')
    ax.plot(x, data_large, label='large cell\n' + r'($30 \times 30 \times 40 \AA^3 $, 1674 atoms)')
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              frameon=False,
              fontsize=10)

    ax.set_xlim(0, 10)
    ax.set_xlabel('Time step (ps)')
    ax.set_ylabel('Potential energy (eV/atom)')

    fig.tight_layout()
    fig.savefig('result.png')
    fig.savefig('result.pdf')


def main():
    data_original = read_data('log_original.lammps')
    data_large = read_data('log_largecell.lammps')
    plot(data_original, data_large)


if __name__ == '__main__':
    main()
