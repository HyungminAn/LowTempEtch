#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="postprocess"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/00_inputs/plot/EffectiveAds/cmd.sh"
dst="plot/EffectiveAds"

target="/data2/andynn/LowTempEtch/07_Equil_LayerThickness"
cd ${target}

for i in $(ls -d {03..07}*/results/*/);do
    cd ${i}
    mkdir -p ${dst}
    cd ${dst}
    echo $(pwd)
    sh ${path_src}

    cd ${target}
done
