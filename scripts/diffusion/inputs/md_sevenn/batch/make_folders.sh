for i in $(seq 1 1 5);do
    mkdir -p "${i}"
    cp lammps.in input.data ${i}
done
