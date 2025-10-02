import matplotlib.pyplot as plt
import pickle


def main():
    color_dict = {
        "blue": (9/255, 115/255, 179/255),     # 파랑
        "orange": (212/255, 98/255, 39/255),  # 주황
        "green": (29/255, 158/255, 116/255),  # 초록
        "pink": (203/255, 121/255, 168/255),  # 분홍
        "yellow": (230/255, 160/255, 36/255)  # 노랑
    }
    with open('vanilla_short/bond_length_dict_NH4.pkl', 'rb') as f:
        v_nh4_bond_length_dict = pickle.load(f)
    with open('vanilla_short/bond_length_dict_SiF6.pkl', 'rb') as f:
        v_sif6_bond_length_dict = pickle.load(f)
    with open('baseline/bond_length_dict_NH4.pkl', 'rb') as f:
        b_nh4_bond_length_dict = pickle.load(f)
    with open('baseline/bond_length_dict_SiF6.pkl', 'rb') as f:
        b_sif6_bond_length_dict = pickle.load(f)

    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    fig, (ax_SiF, ax_NH) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)

    # SiF6
    x = [i * 0.001 for i in range(len(list(b_sif6_bond_length_dict.values())[0]))]
    for y in b_sif6_bond_length_dict.values():
        ax_SiF.plot(x, y,
                    linewidth=1.5,
                    color='tab:grey',
                    label='Scratch NNP')
    x = [i * 0.001 for i in range(len(list(v_sif6_bond_length_dict.values())[0]))]
    for y in v_sif6_bond_length_dict.values():
        ax_SiF.plot(x, y,
                    linewidth=1.5,
                    color=color_dict['orange'],
                    label='Fine-tuned NNP')

    ax_SiF.set_xlim(0, 1)
    ax_SiF.set_ylabel('Si-F\n Bond length\n' + r'($\mathrm{\AA}$)')

    x = [i * 0.001 for i in range(len(list(b_nh4_bond_length_dict.values())[0]))]
    for y in b_nh4_bond_length_dict.values():
        ax_NH.plot(x, y,
                   linewidth=1.5,
                   color='tab:grey',
                   label='Scratch NNP')
    x = [i * 0.001 for i in range(len(list(v_nh4_bond_length_dict.values())[0]))]
    for y in v_nh4_bond_length_dict.values():
        ax_NH.plot(x, y,
                   linewidth=1.5,
                   color=color_dict['orange'],
                   label='Fine-tuned NNP')

    ax_NH.set_xlim(0, 1)
    ax_NH.set_xlabel('Time (ps)')
    ax_NH.set_ylabel('N-H\n Bond length\n' + r'($\mathrm{\AA}$)')

    handles = []
    labels = []
    for ax in [ax_SiF, ax_NH]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique_labels = dict(zip(labels, handles))
    final_labels = list(unique_labels.keys())
    final_handles = list(unique_labels.values())

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.legend(final_handles, final_labels,
               loc='lower center',
               ncol=1,
               fontsize=10,
               frameon=False)

    fig.savefig('result.pdf', dpi=200)
    fig.savefig('result.png')


if __name__ == '__main__':
    main()
