from utils.log import log_function_call  # noqa: E402
from adsorption.lammps.insertion import run as run_insertion_LAMMPS  # noqa: E402
from adsorption.ase.relax import run as run_relaxation_ASE  # noqa: E402
from adsorption.ase.run_reconst import run as run_relaxation_ASE_reconst  # noqa: E402
from adsorption.check.etchant_dissociation import check_etchant_dissociation  # noqa: E402
from adsorption.check.reconstruction import check_reconstruction  # noqa: E402
from adsorption.check.reconstruction import check_reconstruction_reverse  # noqa: E402
from adsorption.remove import remove_etchant_molecule  # noqa: E402


@log_function_call
def repeat_adsorption(output, **inputs):
    '''
    Relax the structure with a etchant molecule:
        1) insert molecule at random position on slab
        2) relaxation of the structure
        3) Check for the dissociation of the etchant molecule
           The number of dissociation --> *chemisorption_ratio*
        4) Check for surface reconstruction
        5) For reconstructed structures, Remove the inserted etchant molecule
        6) Re-relax for the new slab structure
        7) Re-check for the slab reconstruction
    '''
    key = "etchant"
    inputs[key]["paths"]["slab"] =\
        inputs[key]["paths"]["slab"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_1"] =\
        inputs[key]["paths"]["dst_1"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_2"] =\
        inputs[key]["paths"]["dst_2"] % {'dst': inputs['dst']}
    inputs[key]["paths"]["dst_3"] =\
        inputs[key]["paths"]["dst_3"] % {'dst': inputs['dst']}

    run_insertion_LAMMPS(key, **inputs)
    run_relaxation_ASE(key, **inputs)

    idx_etchant_dict = check_etchant_dissociation(key, **inputs)
    idx_reconst_dict = check_reconstruction(key, idx_etchant_dict, **inputs)

    remove_etchant_molecule(key, idx_reconst_dict, **inputs)
    run_relaxation_ASE_reconst(key, idx_reconst_dict, **inputs)

    exclude_dict = check_reconstruction_reverse(
            key, idx_reconst_dict, **inputs)
    phys_dict = {**idx_etchant_dict, **idx_reconst_dict}

    output["exclude_dict"] = exclude_dict
    output["phys_dict"] = phys_dict
