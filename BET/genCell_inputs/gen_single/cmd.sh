#!/bin/bash

function write_inp() {
# Run the first Python script
# input: write_inp.py input.yaml LOGFILE
    local path_src=$1
    local path_input_yaml=$2
    local LOGFILE=$3

    python ${path_src} ${path_input_yaml}  | tee -a ${LOGFILE}
    exit_status=${PIPESTATUS[0]}
    if [ $exit_status -ne 0 ]; then
        echo "Error in writing packmol input"  | tee -a ${LOGFILE}
        return 1
    fi

    return 0
}

function run_packmol() {
# Run packmol
# input: packmol my_input.inp LOGFILE
    local path_src=$1
    local path_input=$2
    local LOGFILE=$3

    ${path_src} < ${path_input}  | tee -a ${LOGFILE}
    exit_status=${PIPESTATUS[0]}
    if [ $exit_status -ne 0 ]; then
        echo "Error in running packmol"  | tee -a ${LOGFILE}
        return 1
    fi

    return 0
}

function merge() {
# Run the merge Python script
# input: merge.py POSCAR my_output.xyz LOGFILE
# output: my_output.xyz
    local path_src=$1
    local path_poscar=$2
    local LOGFILE=$3

    python ${path_src} ${path_poscar} my_output.xyz  | tee -a ${LOGFILE}
    exit_status=${PIPESTATUS[0]}
    if [ $exit_status -ne 0 ]; then
        echo "Error in merging packmol output"  | tee -a ${LOGFILE}
        return 1
    fi

    return 0
}

function main() {
# Define the logfile
    LOGFILE="./script.log"

# Define the paths
    path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/01_genCell/inputs"
    path_src_1="${path_src}/write_inp.py"
    path_input_yaml="./input.yaml"
    path_src_2="/home/andynn/packmol-20.14.4-docs1/packmol"
    path_input="./my_input.inp"
    path_src_3="${path_src}/merge.py"
    path_poscar="${path_src}/POSCAR"

# Write the packmol input
    write_inp ${path_src_1} ${path_input_yaml} ${LOGFILE}
    if [ $? -ne 0 ]; then
        return 1
    fi
# Run packmol
    run_packmol ${path_src_2} ${path_input} ${LOGFILE}
    if [ $? -ne 0 ]; then
        return 1
    fi
# Merge the packmol output into the POSCAR_slab
    merge ${path_src_3} ${path_poscar} ${LOGFILE}
    if [ $? -ne 0 ]; then
        return 1
    fi

    echo "###################################"  | tee -a ${LOGFILE}
    echo "#   Run completed successfully    #"  | tee -a ${LOGFILE}
    echo "# $(date) #" | tee -a ${LOGFILE}
    echo "###################################"  | tee -a ${LOGFILE}
}

main
