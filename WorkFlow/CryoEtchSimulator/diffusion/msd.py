from pathlib import Path
import time

import numpy as np

from ase.io import read
from ase.geometry import get_distances


class MeanSquaredDisplacementPlotter:
    def __init__(self, method, path, step=10, atom_type='F', d_SiF=2.0, d_HF=1.1):
        self.method = method
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"File not found: {self.path}")
        self.dst = self.path.parent

        self.step = step
        self.atom_type = atom_type
        self.d_SiF = d_SiF
        self.d_HF = d_HF

        self.positions = None
        self.mask_F_SiF = None
        self.mask_F_HF = None
        self.mask_F_other = None

    def run(self):
        start_time = time.time()

        self._load_data()
        self._calculate_msd()

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} sec")

    def _load_data(self):
        atoms_format = 'vasp-out' if self.method == 'AIMD' else 'lammps-dump-text'
        atoms_list = read(self.path, index=f'::{self.step}', format=atoms_format)
        initial_image = atoms_list[0]
        idx_F = np.array([atom.index for atom in initial_image if atom.symbol == self.atom_type])

        self.mask_F_SiF, self.mask_F_HF, self.mask_F_other =\
            self._separate_F_index(initial_image)
        self.positions = np.array([atoms[idx_F].get_positions()[:, 0:2] for atoms in atoms_list])

    def _separate_F_index(self, poscar):
        pos = poscar.get_positions()
        _, D_len = get_distances(pos, cell=poscar.get_cell(), pbc=True)
        idx_F = [atom.index for atom in poscar if atom.symbol == 'F']
        idx_Si = [atom.index for atom in poscar if atom.symbol == 'Si']
        idx_H = [atom.index for atom in poscar if atom.symbol == 'H']

        idx_F_SiF = [i for i in idx_F if np.min(D_len[i, idx_Si]) < self.d_SiF]
        idx_F_HF = [i for i in idx_F if np.min(D_len[i, idx_H]) < self.d_HF]
        idx_F_other = [i for i in idx_F if i not in idx_F_SiF + idx_F_HF]

        mask_F_SiF = np.array([i in idx_F_SiF for i in idx_F])
        mask_F_HF = np.array([i in idx_F_HF for i in idx_F])
        mask_F_other = np.array([i in idx_F_other for i in idx_F])

        return mask_F_SiF, mask_F_HF, mask_F_other

    def _calculate_msd(self):
        N = len(self.positions)
        MSD = []

        for delt in range(1, N):
            diffs = self.positions[delt:] - self.positions[:-delt]
            squared_distances = np.sum(diffs**2, axis=-1)

            msd_data = self._calculate_msd_stats(squared_distances)
            MSD.append((delt,) + msd_data)

            self._print_msd_line(delt, msd_data)

        self._save_msd_values(MSD)

    def _calculate_msd_stats(self, squared_distances):
        sd_F_SiF = squared_distances[:, self.mask_F_SiF]
        sd_F_HF = squared_distances[:, self.mask_F_HF]
        sd_F_other = squared_distances[:, self.mask_F_other]

        return (
            np.mean(squared_distances), np.std(squared_distances),
            np.mean(sd_F_SiF), np.std(sd_F_SiF),
            np.mean(sd_F_HF), np.std(sd_F_HF),
            np.mean(sd_F_other), np.std(sd_F_other)
        )

    def _print_msd_line(self, delt, msd_data):
        msd_avg, msd_std, avg_SiF, msd_SiF, avg_HF, msd_HF, avg_other, msd_other = msd_data
        line = f"delt * step: {delt * self.step:5d} "
        line += f"msd_avg: {msd_avg:10.5f} msd_std: {msd_std:10.5f} "
        line += f"avg_SiF: {avg_SiF:10.5f} msd_SiF: {msd_SiF:10.5f} "
        line += f"avg_HF: {avg_HF:10.5f} msd_HF: {msd_HF:10.5f} "
        line += f"avg_other: {avg_other:10.5f} msd_oth: {msd_other:10.5f}"
        print(line)

    def _save_msd_values(self, MSD):
        with open(self.dst/'msd_avg.dat', 'w') as f:
            for dat in MSD:
                line = f"{dat[0] * self.step} " + " ".join(f"{v:10.5f}" for v in dat[1:]) + "\n"
                f.write(line)
