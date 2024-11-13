src="/data2/andynn/LowTempEtch/06_diffusion/03_AFS/02_7net/01_gen_Cell/results"
for i in $(ls -d ${src}/*/);do
    if [ ! -f ${i}/POSCAR_merged ];then
        continue
    fi

    dst=$(basename $i)
    if [ -d ${dst} ];then
        continue
    fi

    mkdir -p ${dst}
    cp ${i}/POSCAR_merged ${dst}/POSCAR
    cp input.yaml ${dst}
    echo "Created ${dst}"
done
