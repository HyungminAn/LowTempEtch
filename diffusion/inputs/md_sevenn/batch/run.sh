cwd=$(pwd)
method="ewcRep"
cell_type="Large"
md_time="10ps"
path_input_data="/data2/andynn/LowTempEtch/04_diffusion/AFS/chgTot/inputs"
for T in 250 300 350 400;do
# for T in 250 400;do
    dst="${cell_type}Cell_IF5_1ML_HF_1ML_${T}K_${md_time}"
    mkdir -p ${dst}

    for file in make_folders.sh run_msd.sh;do
        cp ${cwd}/${file} ${dst}
    done

    if [ ${cell_type} == "Small" ];then
        cp ${path_input_data}/SmallCell_IF5_1ML_HF_1ML.data ${dst}/input.data
    elif [ ${cell_type} == "Large" ];then
        cp ${path_input_data}/LargeCell_IF5_1ML_HF_1ML.data ${dst}/input.data
    fi

    job_name="${method}_${cell_type}${T}K${md_time}"
    sed "s/JOBTITLE/${job_name}/g" batch.j > ${dst}/batch.j

    sed "/250/s/250/${T}/g" lammps.in > ${dst}/lammps.in

    cd ${dst}
    sh make_folders.sh

    # for i in $(ls -d */);do
    #     cd ${i}
    #     sbatch batch.j
    #     cd ..
    # done

    sbatch batch.j

    cd ${cwd}
done
