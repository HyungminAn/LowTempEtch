#!/bin/bash
#SBATCH --nodelist=n008
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --job-name="relax"
#SBATCH --time=04-00:00                 # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out             # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err             # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT     # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# To use gpu
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=5
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

################################################################################
#                                  Path_codes                                  #
################################################################################
path_lmp_bin="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn_d3"
path_lmp_input="lammps.in"
path_relax="/data2/andynn/LowTempEtch/00_codes/relax_gnn_d3.py"
path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/00_inputs/"
path_write_lmp_input="${path_src}/write_lmp_input.py"
path_pos2lmp="${path_src}/pos2lmp.py"
path_lmpdat2vasp="${path_src}/lmpdat2vasp.py"
path_remove_gas="${path_src}/remove_gas.py"

################################################################################
#                              1. Insert molecule                              #
################################################################################
function insert_molecule() {
    local path_GenPoscar=$1

    mkdir -p ${path_GenPoscar}
    cd ${path_GenPoscar}
    #TODO: Generate POSCARs using PACKMOL code

    cd ${cwd}
}

################################################################################
#                              2. Relax Slab + Mol                             #
################################################################################
function relax_slabMol() {
    local path_GenPoscar=$1
    local path_SlabMol=$2
    local n_run=$3

    mkdir -p ${path_SlabMol}
    local cwd=$(pwd)
    for i in $(seq 1 1 ${n_run});do
        local dst="${path_SlabMol}/${i}"
        mkdir -p ${dst}
        python ${path_lmpdat2vasp} ${path_GenPoscar}/${i}.coo
        mv POSCAR ${dst}
        cp input.yaml ${dst}

        cd ${dst}
        python ${path_relax} POSCAR input.yaml
        cd ${cwd}
    done
}

################################################################################
#                              3. Relax Slab only                              #
################################################################################
function relax_slab_only() {
    local path_SlabMol=$1
    local path_SlabOnly=$2
    local n_run=$3
    local gas_name=$4

    mkdir -p ${path_SlabOnly}
    local cwd=$(pwd)
    for i in $(seq 1 1 ${n_run});do
        local dst="${path_SlabOnly}/${i}"
        mkdir -p ${dst}
        local path_poscar_ads="${path_SlabMol}/${i}/POSCAR_relaxed"
        local path_poscar_out="${dst}/POSCAR"
        python ${path_remove_gas} ${path_poscar_ads} ${gas_name} ${path_poscar_out}
        cp input.yaml ${dst}

        cd ${dst}
        python ${path_relax} POSCAR input.yaml
        cd ${cwd}
    done
}

function main() {
################################################################################
#                                    Inputs                                    #
################################################################################
    local gas_name="IF5"
    local n_run=1
    local path_slab="/data2/andynn/LowTempEtch/02_slab/02_AFS/02_7net/POSCAR_relaxed"

    local cwd=$(pwd)
    local path_GenPoscar="${cwd}/gen_poscar"
    local path_SlabMol="${cwd}/run"
    local path_SlabOnly="${cwd}/rerun"


}
