import os
import yaml
import shutil


def read_data_as_dict(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    path_src = "/data2/andynn/LowTempEtch/07_Equil_LayerThickness/01_genCell/results"
    mol_list = [
        i for i in os.listdir(path_src)
        if os.path.isdir(os.path.join(path_src, i))]
    os.makedirs("./results", exist_ok=True)

    path_n_layer_dict = "./n_layer.dat"
    n_layer_dict = read_data_as_dict(path_n_layer_dict)

    for mol in mol_list:
        path_mol = os.path.join(path_src, mol)
        dst = f"./results/{mol}"
        os.makedirs(dst, exist_ok=True)

        shutil.copy(f"{path_mol}/mol.xyz", f"{dst}/mol.xyz")

        with open(f"{path_mol}/result.yaml", "r") as f:
            data = yaml.safe_load(f)

        data['params']['layer_ADDI'] = n_layer_dict['ADDI'][mol]
        data['params']['n_HF'] = 20
        data['params']['mol_size_HF'] = 1.5
        data['params']['layer_HF'] = n_layer_dict['HF'][mol]

        data['path']['path_HF'] = "/data2/andynn/LowTempEtch/07_Equil_LayerThickness/01_genCell/prev_results/HF/mol.xyz"

        with open((f"{dst}/input.yaml"), "w") as f:
            yaml.dump(data, f)

        print(f"Done {mol}")


if __name__ == "__main__":
    main()
