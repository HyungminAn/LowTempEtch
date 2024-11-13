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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

path_lmp="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn_d3"
path_input="lammps.in"

conda activate pub_sevenn

${path_lmp} -in ${path_input}
