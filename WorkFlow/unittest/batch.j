#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu          # Partition name (skylake)
##
#SBATCH --job-name="test"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# rm -rf STD*
path_py="/data2/andynn/LowTempEtch/00_codes/WorkFlow/CryoEtchSimulator/main.py"
python ${path_py} ../CryoEtchSimulator/input.yaml
