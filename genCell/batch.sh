#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="genCell"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

src="/data2/andynn/LowTempEtch/03_gases/benchmark/chgTot"
path_py="/data2/andynn/LowTempEtch/00_codes/genCell/genCell.py"
path_yaml="/data2/andynn/LowTempEtch/00_codes/genCell/input.yaml"
cwd=$(pwd)
for mol in $(ls -d ${src}/*/);do
    mol=$(basename ${mol})
    dst=run/${mol}
    mkdir -p ${dst}
    sed "s|AsF5|${mol}|g" ${path_yaml} > ${dst}/input.yaml

    cd ${dst}
    echo Starting ${dst}
    python ${path_py} input.yaml
    echo Finished ${dst}
    cd ${cwd}
done
