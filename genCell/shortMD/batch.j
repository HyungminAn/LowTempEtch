#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="testMD"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# ### To choose GPU nodes, turn on the option below...
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=5

path_pot="/data2/shared_data/pretrained/7net_chgTot/deployed_serial.pt"
path_lmp="/home/andynn/lammps_sevenn/build/lmp"
path_lmp_in='lammps.in'
${path_lmp} -in ${path_lmp_in} -v path_potential ${path_pot} -v SEEDS ${RANDOM} > lammps.out
