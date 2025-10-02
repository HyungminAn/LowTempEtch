import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot.py <datafile>")
        sys.exit(1)

    dat = np.loadtxt(sys.argv[1])
    md_step, msd_avg, msd_std = dat[:, 0], dat[:, 1], dat[:, 2]

    step_per_image = 100
    time_step = 0.001  # ps
    x = [i * step_per_image * time_step for i in md_step]
    y = msd_avg

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    # ax.errorbar(step, msd_avg, yerr=msd_std, fmt='o', label='MSD')
    ax.plot(x, y, 'b-', alpha=0.5)

    # Get the trend line, y = mx (least square fit)
    # Using np.linalg.lstsq
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax.plot(
        x, m * np.array(x) + c,
        'grey',
        linestyle='--',
        label=f'Fit: {m:.2f}x + {c:.2f}')

    dim = 2  # 2D
    A2_to_cm2 = 1E-16
    ps_to_s = 1E-12
    Diff_coeff = m * A2_to_cm2 / ps_to_s / (2 * dim)  # cm^2/s
    ax.set_title(f'$D = {Diff_coeff:.3e} cm^2/s$')

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD ($A^2$)')
    ax.legend()

    fig.tight_layout()
    fig.savefig('msd.png')


if __name__ == '__main__':
    main()
