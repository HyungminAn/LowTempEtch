from ase.io import read
from ase.geometry import get_distances
from ase.neighborlist import NeighborList
from ase.neighborlist import natural_cutoffs


def is_surface_reconstructed(path_dump, idx_HF, change_ratio_cutoff=1.2):

    poscar_before_rlx = read(path_dump, index=0)
    poscar_after_rlx = read(path_dump, index=-1)

    pos_before = poscar_before_rlx.get_positions()
    pos_after = poscar_after_rlx.get_positions()
    cell = poscar_before_rlx.get_cell()

    _, D_before = get_distances(pos_before, cell=cell, pbc=True)
    _, D_after = get_distances(pos_after, cell=cell, pbc=True)

    cn_mats = []

    for poscar in [poscar_before_rlx, poscar_after_rlx]:
        cutoffs = natural_cutoffs(poscar)
        nl = NeighborList(cutoffs, self_interaction=False)
        nl.update(poscar)

        cn_mat = nl.get_connectivity_matrix()
        cn_mats.append(cn_mat)

    cn_mat_before, cn_mat_after = cn_mats
    cn_diff = cn_mat_after - cn_mat_before

    rows, cols = cn_diff.nonzero()
    for row, col in zip(rows, cols):
        if row in idx_HF or col in idx_HF:
            continue

        d_before = D_before[row, col]
        d_after = D_after[row, col]
        change_ratio = max(d_before, d_after) / min(d_before, d_after)

        if change_ratio > change_ratio_cutoff:
            return True

    return False
