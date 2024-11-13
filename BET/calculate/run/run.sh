path_src="/data2/andynn/LowTempEtch/07_Equil_LayerThickness/08_Calculate/main.py"
cwd=$(pwd)
# for i in $(ls -d */);do
for i in XeF2;do
    cd ${i}
    python ${path_src} input.yaml
    cd ${cwd}
done
