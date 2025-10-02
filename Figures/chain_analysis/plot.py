import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    with open('IF5_1ML_HF_1ML.pickle', 'rb') as f:
        data_IF5_HF = pickle.load(f)
    with open('HF_1ML.pickle', 'rb') as f:
        data_HF = pickle.load(f)

    # cm = 1/2.54
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        })
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    x = np.array([i for i in range(len(data_IF5_HF))]) * 0.1
    y1 = np.array(data_IF5_HF)
    y2 = np.array(data_HF)

    ax.plot(x, y2, label='AFS slab', color='#4fb696')
    ax.plot(x, y1, label='AFS slab + IF$_5$ 1ML', color='#ca7baa')

    ax.set_xlim(0, 10)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(r'Average length of HF chain ($\mathrm{\AA}$)')

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fontsize=10,
              frameon=False)

    fig.tight_layout()
    fig.savefig('result.png')
    fig.savefig('result.pdf')


if __name__ == '__main__':
    main()
