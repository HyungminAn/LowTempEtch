#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="JOBTITLE"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# ### To choose GPU nodes, turn on the option below...
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=5

# path_potential='/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTune/pot/vanilla/deployed_serial.pt'
# path_potential='/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTune/pot/ewc/deployed_serial.pt'
# path_potential='/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTune/pot/ewc_replay/deployed_serial.pt'

# path_potential="/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTuneVer2/pot/vanilla/deployed_serial.pt"
# path_potential="/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTuneVer2/pot/ewc/deployed_serial.pt"
path_potential="/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTuneVer2/pot/ewc_replay//deployed_serial.pt"

path_lmp_in='lammps.in'

cwd=$(pwd)
for i in $(seq 1 1 5);do
    cd ${i}
    lmp_sevenn_d3 -in ${path_lmp_in} -v SEEDS ${RANDOM} -v path_potential ${path_potential} > lammps.out
    cd ${cwd}
done
