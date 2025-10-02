import numpy as np
import matplotlib.pyplot as plt


def plot(E_dict, **inputs):
    '''
    plot *energy.dat*
    '''
    e = np.array([i for i in E_dict.values()])

    x_min = inputs["plot"]["x_min"]
    x_max = inputs["plot"]["x_max"]
    if not x_min and not x_max:
        x_min = min(e)
        x_max = max(e)

    name = inputs["additive"]["mol_info"]["name"]
    prop_dict = {
        'range': (x_min, x_max),
        'bins': 30,
        'alpha': 0.3,
        'color': "green",
        "label": f'{name} ({len(e)})',
    }

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()

    ax.hist(e,  **prop_dict)
    ax.set_xlabel('$E_{ads}$ (eV)')
    ax.set_ylabel('Counts')
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('result.png')
