#!/bin/bash
# Usage : sh get_energy.sh *folder_name* *gas_name*

##############################################################################
#                 Last modified: 2024. 07. 10 (Hyungmin An)                  #
##############################################################################

##############################################################################
#                                   Inputs                                   #
##############################################################################
path_src="$1"
mol_name="$2"
if [ ${path_src} == "" ];then
    echo "Please input the source path."
    exit
fi
if [ ${mol_name} == "" ];then
    echo "Please input the molecule name."
    exit
fi

LOGFILE="energy.dat"
if [ ${mol_name} == "HF" ];then
    path_mol="/data2/andynn/LowTempEtch/03_gases/run/base/HF/02_SevenNet/lammps.out"
else
    path_mol="/data2/andynn/LowTempEtch/03_gases/run/${mol_name}/02_SevenNet/lammps.out"
fi

##############################################################################
#                                Get E_before                                #
##############################################################################
E_mol=$(grep -A 1 'Energy initial' ${path_mol} | tail -n 1 | awk '{print $3}')

##############################################################################
#                                Get E_after                                 #
##############################################################################
cat /dev/null > ${LOGFILE}
# Get the sorted paths by their base names
sorted_paths=$(ls -d ${path_src}/run/* | awk -F/ '{print $NF ":" $0}' | sort -t: -k1,1n | cut -d: -f2)

# Process each sorted path
for i in $sorted_paths; do
    i=$(basename $i)
    path_src_before="${path_src}/rerun/${i}"
    path_src_after="${path_src}/run/${i}"

    # Check whether the calculation is done
    if [[ ! -e ${path_src_before}/log || $(grep "terminated" ${path_src_before}/log | wc -l) -eq 0 ]]; then
        echo "${path_src_before} is not terminated."
        continue
    elif [[ ! -e ${path_src_after}/log || $(grep "terminated" ${path_src_after}/log | wc -l) -eq 0 ]]; then
        echo "${path_src_after} is not terminated."
        continue
    fi

    src="${path_src_before}/thermo.dat"
    E_before=$(awk '{print $2}' ${src} | tail -n 1)
    E_before=$(echo "$E_before + $E_mol" | bc -l)

    src="${path_src_after}/thermo.dat"
    E_after=$(awk '{print $2}' ${src} | tail -n 1)

    E_phys=$(echo "${E_after} - ${E_before}" | bc -l)

    echo ${i} : ${E_phys} eV >> ${LOGFILE}
    echo "${i} Done"
done
