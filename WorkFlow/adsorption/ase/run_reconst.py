from pathlib import Path

from utils.log import log_function_call  # noqa: E402
from relax.relax_sevenn_d3 import relax  # noqa: E402


@log_function_call
def run(key, idx_reconst_dict, **inputs):
    '''
    run_relaxation_ASE function for reconstructed slabs
    '''
    path_dst = inputs[key]["paths"]["dst_3"]
    for i in idx_reconst_dict.keys():
        dst = f'{path_dst}/{i}'
        dst_poscar = f'{dst}/POSCAR'

        p = Path(dst)
        poscar_relaxed = p / inputs["relax"]["path"]["output"]
        if poscar_relaxed.exists():
            continue

        is_relax_success = relax(dst_poscar, dst, **inputs)
