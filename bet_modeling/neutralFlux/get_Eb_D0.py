import matplotlib.pyplot as plt
import numpy as np


def main():

    kB = 8.617E-05  # eV/K
    dat = [
        (200, 1.090E-05),
        (230, 1.581E-05),
        (270, 4.353E-04),
        # (300, 9.791E-04),
    ]


    x = []
    y = []
    for (temp, diff_coeff) in dat:
        x.append(1/(kB*temp))
        y.append(np.log(diff_coeff))
    x = np.array(x)
    y = np.array(y)

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    prop_dict_scatter = {
        's': 100,
        'color': 'black',
    }
    ax.scatter(x, y, **prop_dict_scatter)

    # add trend line
    m, b = np.polyfit(x, y, 1)

    y_pred = m * x + b
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)

    title = f'$E_a$ = {-m:.3f} eV\n'
    title += f'$D_0$ = {np.exp(b):.2e} $cm^2/s$'
    ax.set_title(title)
    prop_dict_line = {
        'linestyle': '--',
        'color': 'grey',
        'label': f'$R^2$ = {r_squared:.4f}',
    }
    ax.plot(x, y_pred, **prop_dict_line)

    ax.set_xlabel('$1/(k_{B}T)$')
    ax.set_ylabel('$ln(D)$')
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('Eb_D0.png')


if __name__ == '__main__':
    main()
