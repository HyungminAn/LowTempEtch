#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="msd"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

cwd=$(pwd)
src="/data2/andynn/LowTempEtch/04_diffusion/codes/msd/cmd.sh"
for i in $(ls -d Large*{250,300,350,400}K*10ps/*/);do
    if [ ! -f "${i}/FINAL.coo" ];then
        continue
    fi

    if [ -f "${i}/msd/msd.png" ];then
        continue
    fi

    mkdir -p ${i}/msd
    cd ${i}/msd
    sh ${src} ../dump.lammps MD
    cd ${cwd}

    echo "Done: ${i}"
done
