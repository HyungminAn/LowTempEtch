#!/bin/bash
#SBATCH --nodelist=n014
#SBATCH --partition=gpu2
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --job-name="relax"
#SBATCH --time=04-00:00                 # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out             # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err             # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT     # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# To use gpu
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# path_lmp="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn_d3"
# path_input="lammps.in"

path_relax="/data2/andynn/LowTempEtch/00_codes/relax_gnn_d3.py"

for i in $(find . -maxdepth 1 -type d ! -name '.' -exec ls -d {} \;);do
    cd ${i}

    if [ ! -f "POSCAR_relaxed" ];then
        python ${path_relax} POSCAR input.yaml
    fi

    cd ..
done
