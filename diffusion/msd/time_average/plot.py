import sys
import numpy as np
import matplotlib.pyplot as plt


def get_trend_line(x, y, **fit_info):
    '''
    Get the trend line, y = mx (least square fit)
    Using np.linalg.lstsq
    '''
    trunc_init, trunc_end = fit_info['trunc_ratio']
    x_init, x_end = int(len(x) * trunc_init), int(len(x) * trunc_end)
    x_trunc = x[x_init:x_end]
    y_init, y_end = int(len(y) * trunc_init), int(len(y) * trunc_end)
    y_trunc = y[y_init:y_end]
    A = np.vstack([x_trunc, np.ones(len(x_trunc))]).T
    slope, intercept = np.linalg.lstsq(A, y_trunc, rcond=None)[0]
    x_fit = x_trunc
    y_fit = slope * np.array(x_trunc) + intercept
    return x_fit, y_fit, slope, intercept


def plot(dat_type, x, y, ax, **fit_info):
    dim = 2
    A2_to_cm2 = 1E-16
    ps_to_s = 1E-12

    # ax.errorbar(step, msd_avg, yerr=msd_std, fmt='o', label='MSD')
    ax.plot(x, y, color=fit_info['color'], alpha=0.5, label=dat_type)

    x_fit, y_fit, slope, intercept = get_trend_line(x, y, **fit_info)
    Diff_coeff = slope * A2_to_cm2 / ps_to_s / (2 * dim)  # cm^2/s
    label = f'Fit: {slope:.2f}x + {intercept:.2f}\n'
    label += f'$D = {Diff_coeff:.3e} cm^2/s$'
    ax.plot(
        x_fit, y_fit, label=label,
        color=fit_info['color'], **fit_info['plot_options'])


def main():
    if len(sys.argv) != 3:
        print("Usage: python plot.py <datafile> <method>")
        sys.exit(1)
    path_to_data = sys.argv[1]
    method = sys.argv[2]

    md_info = {
        'step_per_image': 100 if method == 'MD' else 1,
        'time_step': 0.001,  # ps unit
    }

    fit_info = {
        'trunc_ratio': (0.1, 0.7),
        'plot_options': {
            'linestyle': '--',
        },
        'color': None,
    }

    dat = np.loadtxt(path_to_data)
    md_step, msd_avg, _ = dat[:, 0], dat[:, 1], dat[:, 2]
    msd_avg_SiF, _ = dat[:, 3], dat[:, 4]
    msd_avg_HF, _ = dat[:, 5], dat[:, 6]
    msd_avg_others, _ = dat[:, 7], dat[:, 8]

    x = [i * md_info['step_per_image'] * md_info['time_step'] for i in md_step]
    y = msd_avg

    data_to_plot = [
        ('Total', msd_avg, 'black'),
        ('SiF', msd_avg_SiF, 'green'),
        ('HF', msd_avg_HF, 'red'),
        ('Others', msd_avg_others, 'blue'),
    ]

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12, 6))

    for dat_type, y, color in data_to_plot:
        fit_info['color'] = color
        plot(dat_type, x, y, ax, **fit_info)

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD ($A^2$)')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.tight_layout()
    fig.savefig('msd.png')


if __name__ == '__main__':
    main()
