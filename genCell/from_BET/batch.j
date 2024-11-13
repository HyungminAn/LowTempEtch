#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=andynn          # Partition name (skylake)
##
#SBATCH --job-name="packmol"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/01_genCell/inputs/main.py"
# for i in $(ls -d */);do
for i in AsF5;do
    cd $i
    python ${path_src} input.yaml
    cd ..
done
