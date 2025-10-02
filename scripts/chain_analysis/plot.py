import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    with open('IF5_1ML_HF_1ML.pickle', 'rb') as f:
        data_IF5_HF = pickle.load(f)
    with open('HF_1ML.pickle', 'rb') as f:
        data_HF = pickle.load(f)

    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(8.5*cm, 8.5*cm))

    x = np.array([i for i in range(len(data_IF5_HF))]) * 0.1
    y1 = np.array(data_IF5_HF)
    y2 = np.array(data_HF)

    ax.plot(x, y1, label='IF5+HF', color='#D76224')
    ax.plot(x, y2, label='HF', color='#0c74b2')

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('average length of HF chain')

    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('chain_analysis.png')


if __name__ == '__main__':
    main()
