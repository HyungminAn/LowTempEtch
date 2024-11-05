import os

from utils.log import log_function_call  # noqa: E402
from adsorption.check_disso import is_etchant_dissociated  # noqa: E402
from adsorption.check_surf_reconst import is_surface_reconstructed  # noqa: E402


@log_function_call
def check_etchant_dissociation(key, **inputs):
    '''
    classify results to physisorption and chemisorption,
    depending on whether the inserted etchant molecule has been dissociated.
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    path_slab = inputs[key]["paths"]["slab"]
    n_repeat = inputs[key]["mol_info"]["n_repeat"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    idx_etchant_dict = {}
    for i in range(n_repeat):
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
            continue
        cond, idx_etchant = is_etchant_dissociated(src, path_slab)
        if cond:
            continue
        idx_etchant_dict[i] = idx_etchant

    return idx_etchant_dict


@log_function_call
def check_reconstruction(key, idx_etchant_dict, **inputs):
    '''
    Check whether there was a reconstruction during structure optimization
    '''
    path_src = inputs[key]["paths"]["dst_2"]
    format_rlx_trj = inputs["relax"]["path_extxyz"]
    idx_reconst = {}
    for i, idx_etchant in idx_etchant_dict.items():
        src = f'{path_src}/{i}/{format_rlx_trj}'
        if not os.path.isfile(src):
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
    format_rlx_trj = inputs["relax"]["path_extxyz"]
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
