import numpy as np
import yaml
import matplotlib.pyplot as plt


def read_dat_as_dict(path_dat):
    with open(path_dat, "r") as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        line = line.strip().split()
        data[int(line[0])] = float(line[1])
    return data


def get_layer_thickness(T_list, p_list, E_dict):
    '''
    Get the equilibrium layer thickness of species A,
    at given temperature T and pressure pA
    '''
    kB = 8.617333262145e-5  # eV/K
    T_grid, p_grid = np.meshgrid(T_list, p_list)
    v_grid = np.zeros_like(T_grid)
    n_row, n_col = p_grid.shape
    for i in range(n_row):
        for j in range(n_col):
            T = T_grid[i, j]
            p = p_grid[i, j]
            E_1A = E_dict["E_1A"][T]
            E_LA = E_dict["E_LA"][T]
            c = np.exp((E_1A - E_LA) / (kB * T))
            print(f"T {T:10.2f}, E_1A {E_1A:10.2f}, E_LA {E_LA:10.2f}, c {c:10.2f}")
            v = c * p / (1 - p) / (1 + p * (c - 1))
            v_grid[i, j] = v

    return T_grid, p_grid, v_grid


def plot(x, y, z, **inputs):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, z, cmap="viridis")

    n_row, n_col = z.shape

    for i in range(0, n_row):
        for j in range(0, n_col):
            # ax.text(x[i][j], y[i][j], 0.0, f"{z[i][j]:.2f}")
            ax.text(x[i][j], y[i][j], z[i][j], f"{z[i][j]:.2f}")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("$P / P0$")
    ax.set_zlabel("Layer thickness (ML)")

    fig.tight_layout()
    fig.savefig("output.png")
    print("Saved output.png")


def main():
    with open('input.yaml') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    path_1A = inputs["path"]["1A"]
    path_LA = inputs["path"]["LA"]

    E_1A_dict = read_dat_as_dict(path_1A)
    E_LA_dict = read_dat_as_dict(path_LA)
    E_dict = {"E_1A": E_1A_dict, "E_LA": E_LA_dict}

    T_min = inputs["temperature"]["min"]
    T_max = inputs["temperature"]["max"]
    T_step = inputs["temperature"]["step"]
    T_list = np.arange(T_min, T_max, T_step)

    p_min = inputs["pressure"]["min"]
    p_max = inputs["pressure"]["max"]
    p_step = inputs["pressure"]["step"]
    p_list = np.arange(p_min, p_max, p_step)

    T_grid, p_grid, v_grid = get_layer_thickness(T_list, p_list, E_dict)
    plot(T_grid, p_grid, v_grid)


if __name__ == "__main__":
    main()
