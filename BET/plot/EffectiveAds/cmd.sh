#!/bin/bash

cwd=$(pwd)
if [ $(basename ${cwd}) != "EffectiveAds" ];then
    echo "Please run this script in the *EffectiveAds* directory"
    exit 1
fi

path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/00_inputs/plot/EffectiveAds/get_energy.sh"
path_py="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/00_inputs/plot/EffectiveAds/plot_E_ads_eff.py"

wd=$(realpath ${cwd}/../../)
echo wd: ${wd}
mol_name=$(basename ${wd})
echo mol_name: ${mol_name}
cal_type=$(realpath ${cwd}/../../../../)
cal_type=$(basename ${cal_type})
echo cal_type: ${cal_type}

if [[ ${cal_type} == 06* ]];then
    mol_name="HF"
fi
echo "Starting ${cal_type}/${mol_name}..."

sh ${path_src} ${wd} ${mol_name}
python ${path_py} energy.dat

echo "...Finished ${cal_type}/${mol_name}"
