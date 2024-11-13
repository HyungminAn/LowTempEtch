import matplotlib.pyplot as plt
import numpy as np

def main():
    md_info = {
        'step_per_image': 100,
        'time_per_step': 0.001,  # ps unit
    }

    plot_info = {
        'figsize': (12, 6),
        'fitting_truncation': 0.1,  # ratio, from beginning
        'prop_dict': {
            'linestyle': '--',
            'alpha': 0.3,
        }
    }
    output = 'msd.out'

    dat = np.loadtxt(output, skiprows=1)
    with open(output, 'r') as f:
        line = f.readline()
        elem_list = line.split()[1:]

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=plot_info['figsize'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x = dat[1:, 0] * md_info['step_per_image'] * md_info['time_per_step']
    for idx, elem in enumerate(elem_list):
        y = dat[1:, 1 + 4 * idx]

        # Get the trend line, y = mx (least square fit)
        # Using np.linalg.lstsq
        trunc_ratio = int(len(x)*plot_info['fitting_truncation'])
        x_fit = x[trunc_ratio:]
        y_fit = y[trunc_ratio:]
        A = np.vstack([x_fit, np.ones(len(x_fit))]).T
        m, c = np.linalg.lstsq(A, y_fit, rcond=None)[0]

        dim = 2  # 2D
        A2_to_cm2 = 1E-16
        ps_to_s = 1E-12
        Diff_coeff = m * A2_to_cm2 / ps_to_s / (2 * dim)  # cm^2/s

        ax.plot(
            x_fit, m * np.array(x_fit) + c,
            c=colors[idx], **(plot_info['prop_dict'])
        )

        label = f'{elem}: $D = {Diff_coeff:.3e} cm^2/s$'
        ax.plot(x, y, label=label, c=colors[idx])

    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD ($A^2$)')

    fig.tight_layout()
    fig.savefig('msd.png')


if __name__ == '__main__':
    main()
