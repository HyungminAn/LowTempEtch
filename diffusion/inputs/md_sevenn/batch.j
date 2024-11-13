#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="md"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

path_lmp="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn_d3"
path_lmp_in='lammps.in'
${path_lmp} -in ${path_lmp_in} -v SEEDS ${RANDOM} > lammps.out
