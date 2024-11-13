if [ $# -ne 2 ]; then
    echo "Usage: cmd.sh <path_to_dump> <MD/AIMD>"
    exit 1
fi
sh /data2/andynn/LowTempEtch/04_diffusion/codes/msd/msd_average/cmd.sh $1 $2
