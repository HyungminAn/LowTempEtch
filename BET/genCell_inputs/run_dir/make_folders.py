import os
import shutil

from ase.io import read, write


def main():
    path_src = "/data2/andynn/LowTempEtch/03_gases/run"
    mol_list = [
        "AsF5",
        "BF3",
        "BiF5",
        "BrF5",
        "C2I2F4",
        "C2IF5",
        "CF3I",
        "ClF5",
        "CoCl2",
        "COF2",
        "COS",
        "CS2",
        "HfCl4",
        "HI",
        "IBr",
        "IF5",
        "IF7",
        "NbF5",
        "NH4F",
        "PF3",
        "PF5",
        "SO2",
        "TaCl4",
        "TaF5",
        "TiCl4",
        "WF6",
        "XeF2",
    ]
    for mol in mol_list:
        src = f"{path_src}/{mol}/02_SevenNet/dump.lammps"
        dst = mol
        os.makedirs(dst, exist_ok=True)
        poscar = read(src)
        write(f"{dst}/mol.xyz", poscar, format="xyz")
        shutil.copy("./input.yaml", f"{dst}/input.yaml")
        print(f"Created folder {dst}")


if __name__ == "__main__":
    main()
