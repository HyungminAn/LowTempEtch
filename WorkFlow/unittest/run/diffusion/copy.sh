src="/data2/andynn/LowTempEtch/04_diffusion/AFS/FineTuneVer2/ewc_replay"
for temp in {250..400..50};do
    for iter in {1..5};do
        dst="${temp}/$((iter-1))"
        mkdir -p ${dst}
        ln -s ${src}/LargeCell_IF5_1ML_HF_1ML_${temp}K_10ps/${iter}/FINAL.coo ${dst}/FINAL.coo
        ln -s ${src}/LargeCell_IF5_1ML_HF_1ML_${temp}K_10ps/${iter}/dump.lammps ${dst}/dump.lammps
    done
done
