from utils.log import log_function_call  # noqa: E402
from adsorption.lammps.insertion import run as run_insertion_LAMMPS  # noqa: E402
from adsorption.ase.relax import run as run_relaxation_ASE  # noqa: E402
from adsorption.select import select_slab_with_minimal_energy  # noqa: E402


@log_function_call
def make_slab_with_additive(output, **inputs):
    '''
    Make a relaxed structure with additive:
        1) insert molecule at random position on slab
        2) relaxation of the structure
        3) select the slab structure with the lowest energy
    '''
    key = "additive"
    inputs[key]["paths"]["dst_1"] =\
        inputs[key]["paths"]["dst_1"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_2"] =\
        inputs[key]["paths"]["dst_2"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_3"] =\
        inputs[key]["paths"]["dst_3"] % {'dst': inputs['dst']}

    run_insertion_LAMMPS(key, **inputs)
    run_relaxation_ASE(key, **inputs)
    select_slab_with_minimal_energy(key, output, **inputs)
