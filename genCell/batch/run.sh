cwd=$(pwd)
path_src="/data2/andynn/LowTempEtch/00_codes/diffusion/genCell/genCell.py"
for i in $(ls -d results/*/);do
    cd ${i}
    python ${path_src} input.yaml
    cd ${cwd}
    echo "Done: ${i}"
done
