path_src="/data2/andynn/LowTempEtch/00_codes/diffusion/msd/time_average/msd.py"
path_plot="/data2/andynn/LowTempEtch/00_codes/diffusion/msd/time_average/plot.py"
path_dump=$1
path_method=$2
if [ -z $path_dump ]; then
    echo "Please provide the path to the dump file."
    exit 1
fi
if [ -z $path_method ]; then
    path_method="MD"
fi
python ${path_src} ${path_method} ${path_dump}
python ${path_plot} msd_avg.dat
