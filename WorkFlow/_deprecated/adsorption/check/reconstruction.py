from pathlib import Path
import os

from ase.io import read
from ase.geometry import get_distances
from ase.neighborlist import NeighborList
from ase.neighborlist import natural_cutoffs

from utils.log import log_function_call  # noqa: E402


@log_function_call
def check_reconstruction(key, idx_etchant_dict, **inputs):
    '''
    Check whether there was a reconstruction during structure optimization
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    format_rlx_trj = inputs["relax"]["path"]["traj"]
    idx_reconst = {}

    for i, idx_etchant in idx_etchant_dict.items():
        src = Path(f'{path_src}/{i}/{format_rlx_trj}')
        if not src.exists():
            continue

        if is_surface_reconstructed(src, idx_etchant):
            idx_reconst[i] = idx_etchant

    for k in idx_reconst.keys():
        idx_etchant_dict.pop(k, None)

    return idx_reconst


@log_function_call
def check_reconstruction_reverse(key, idx_etchant_dict, **inputs):
    '''
    check_reconstruction function in reverse_direction
    '''
    path_src = inputs[key]["paths"]["dst_3"]
    format_rlx_trj = inputs["relax"]["path"]["traj"]
    idx_reconst = {}

    for i, idx_etchant in idx_etchant_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
            continue

        if is_surface_reconstructed(src, []):
            idx_reconst[i] = None

    for k in idx_reconst.keys():
        idx_etchant_dict.pop(k, None)

    return idx_reconst


def get_cn_mat(image):
    cutoffs = natural_cutoffs(image)
    nl = NeighborList(cutoffs, self_interaction=False)
    nl.update(image)

    return nl.get_connectivity_matrix()


def get_dist_mat(image):
    pos = image.get_positions()
    cell = image.get_cell()
    _, D = get_distances(pos, cell=cell, pbc=True)

    return D


def is_surface_reconstructed(path_dump, idx_HF, change_ratio_cutoff=1.2):
    poscar_before_rlx = read(path_dump, index=0)
    poscar_after_rlx = read(path_dump, index=-1)

    D_before = get_dist_mat(poscar_before_rlx)
    D_after = get_dist_mat(poscar_after_rlx)

    cn_mat_before = get_cn_mat(poscar_before_rlx)
    cn_mat_after = get_cn_mat(poscar_after_rlx)
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
