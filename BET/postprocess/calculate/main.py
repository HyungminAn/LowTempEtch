import numpy as np
import yaml
import matplotlib.pyplot as plt


def read_file_as_dict(file_path, type_key=str, type_value=float):
    '''
        Read file format of:
        key1 value1
        key2 value2
        ...

        as {key1: value1, key2: value2, ...}
    '''
    with open(file_path) as f:
        lines = f.readlines()
    result = {}
    for i in lines:
        key, value = i.strip().split()

        key = type_key(key)
        value = type_value(value)

        result[key] = value
    return result


def load_energy(T_list, **inputs):
    '''
        Load effective adsorption energy as a function of temperature
    '''
    eV_to_J = inputs['constants']["eV_to_J"]

    path_E_1A = inputs["path_ads_E"]["species_A"]["1A"]
    path_E_LA_A = inputs["path_ads_E"]["species_A"]["LA_A"]
    path_E_LA_B = inputs["path_ads_E"]["species_A"]["LA_B"]
    path_E_1B = inputs["path_ads_E"]["species_B"]["1B"]
    path_E_LB_A = inputs["path_ads_E"]["species_B"]["LB_A"]
    path_E_LB_B = inputs["path_ads_E"]["species_B"]["LB_B"]

    E_1A = read_file_as_dict(path_E_1A, type_key=int)
    E_LA_A = read_file_as_dict(path_E_LA_A, type_key=int)
    E_LA_B = read_file_as_dict(path_E_LA_B, type_key=int)
    E_1B = read_file_as_dict(path_E_1B, type_key=int)
    E_LB_A = read_file_as_dict(path_E_LB_A, type_key=int)
    E_LB_B = read_file_as_dict(path_E_LB_B, type_key=int)

    E_1A = {T: E_1A[T] * eV_to_J for T in T_list}
    E_LA_A = {T: E_LA_A[T] * eV_to_J for T in T_list}
    E_LA_B = {T: E_LA_B[T] * eV_to_J for T in T_list}
    E_1B = {T: E_1B[T] * eV_to_J for T in T_list}
    E_LB_A = {T: E_LB_A[T] * eV_to_J for T in T_list}
    E_LB_B = {T: E_LB_B[T] * eV_to_J for T in T_list}

    result = {
        "E_1A": E_1A,
        "E_LA_A": E_LA_A,
        "E_LA_B": E_LA_B,
        "E_1B": E_1B,
        "E_LB_A": E_LB_A,
        "E_LB_B": E_LB_B,
    }

    return result


def get_saturated_pressure(T_list, E_dict, **inputs):
    '''
        Get the saturated pressure of each species
        p_0A_sat: saturation pressure of species A at given T
        p_0B_sat: saturation pressure of species B at given T
        p_AA_sat: saturation pressure of species A on species A at given T
        p_AB_sat: saturation pressure of species A on species B at given T
        p_BA_sat: saturation pressure of species B on species A at given T
        p_BB_sat: saturation pressure of species B on species B at given T
    '''

    amu_to_kg = inputs['constants']["amu_to_kg"]
    eV_to_J = inputs['constants']["eV_to_J"]
    Pa_to_mTorr = inputs['constants']["Pa_to_mTorr"]

    v0 = inputs['constants']["v0"]
    kB = inputs['constants']["kB"] * eV_to_J
    mass_dict = read_file_as_dict(inputs["path_mass"])
    name_A = inputs["name"]["species_A"]
    name_B = inputs["name"]["species_B"]
    area_A = inputs["area"]["species_A"]
    area_B = inputs["area"]["species_B"]

    mass_A = mass_dict[name_A] * amu_to_kg
    mass_B = mass_dict[name_B] * amu_to_kg
    K_0A = {T: area_A / (v0 * np.sqrt(2*np.pi*mass_A*kB*T)) for T in T_list}
    K_0B = {T: area_B / (v0 * np.sqrt(2*np.pi*mass_B*kB*T)) for T in T_list}

    def get_p_sat(K, E, T):
        return 1 / (K[T] * np.exp(E[T] / (kB * T))) * Pa_to_mTorr

    p_0A_sat = {T: get_p_sat(K_0A, E_dict["E_1A"], T) for T in T_list}
    p_0B_sat = {T: get_p_sat(K_0B, E_dict["E_1B"], T) for T in T_list}
    p_AA_sat = {T: get_p_sat(K_0A, E_dict["E_LA_A"], T) for T in T_list}
    p_AB_sat = {T: get_p_sat(K_0A, E_dict["E_LA_B"], T) for T in T_list}
    p_BA_sat = {T: get_p_sat(K_0B, E_dict["E_LB_A"], T) for T in T_list}
    p_BB_sat = {T: get_p_sat(K_0B, E_dict["E_LB_B"], T) for T in T_list}

    p_sat_dict = {
        "p_0A_sat": p_0A_sat,
        "p_0B_sat": p_0B_sat,
        "p_AA_sat": p_AA_sat,
        "p_AB_sat": p_AB_sat,
        "p_BA_sat": p_BA_sat,
        "p_BB_sat": p_BB_sat,
    }

    return p_sat_dict


def get_layer_thickness(T, E_dict, p_sat_dict, **inputs):
    '''
    Get the equilibrium layer thickness of species A and B,
    at given temperature T and pressure pA, pB
    '''

    p_0A_sat = p_sat_dict["p_0A_sat"][T]
    p_0B_sat = p_sat_dict["p_0B_sat"][T]
    p_AA_sat = p_sat_dict["p_AA_sat"][T]
    p_AB_sat = p_sat_dict["p_AB_sat"][T]
    p_BA_sat = p_sat_dict["p_BA_sat"][T]
    p_BB_sat = p_sat_dict["p_BB_sat"][T]

    pA_max = min(p_AA_sat, p_AB_sat)
    pB_max = min(p_BA_sat, p_BB_sat)
    pA_ngrid = inputs["process_variables"]["pressure"]["species_A"]["n_grid"]
    pB_ngrid = inputs["process_variables"]["pressure"]["species_B"]["n_grid"]
    pA_list = np.linspace(0, pA_max, pA_ngrid)
    pB_list = np.linspace(0, pB_max, pB_ngrid)

    pA_grid, pB_grid = np.meshgrid(pA_list, pB_list)
    vA_grid, vB_grid = np.zeros_like(pA_grid), np.zeros_like(pB_grid)

    v_init = np.array([1.0, 1.0])
    Id_mat = np.eye(2)

    n_row, n_col = pA_grid.shape
    for i in range(n_row):
        for j in range(n_col):
            pA, pB = pA_grid[i, j], pB_grid[i, j]

            ratio_pA_p0A = pA / p_0A_sat
            ratio_pB_p0B = pB / p_0B_sat

            P0_11, P0_12 = ratio_pA_p0A, 0.0
            P0_21, P0_22 = 0.0, ratio_pB_p0B
            P0 = np.array([
                [P0_11, P0_12],
                [P0_21, P0_22],
            ])

            P_11, P_12 = pA / p_AA_sat, pA / p_AB_sat
            P_21, P_22 = pB / p_BA_sat, pB / p_BB_sat
            P = np.array([
                [P_11, P_12],
                [P_21, P_22],
            ])

            eigenvalues, _ = np.linalg.eig(P)
            magnitudes = np.abs(eigenvalues)
            if np.any(magnitudes >= 1):
                print("Warning: Unstable equilibrium")
                print(f"pA = {pA:.2e}, pB = {pB:.2e}")
                print(f"eigenvalues = {eigenvalues}")
                print(f"magnitudes = {magnitudes}")
                print("#"*79)
                continue

            IsubP = Id_mat - P
            inv = np.linalg.inv(IsubP)
            AB = inv @ P0 @ v_init
            AB_tilde = inv @ AB

            A, B = AB
            A_tilde, B_tilde = AB_tilde

            vA = A_tilde / (1 + A + B)
            vB = B_tilde / (1 + A + B)

            vA_grid[i, j] = vA
            vB_grid[i, j] = vB

    return pA_grid, pB_grid, vA_grid, vB_grid


def plot(dat, figax=(None, None), datalabel=True, **inputs):
    T = dat["T"]
    pA_grid = dat["pA_grid"]
    pB_grid = dat["pB_grid"]
    vA_grid = dat["vA_grid"]
    vB_grid = dat["vB_grid"]

    name_A = inputs["name"]["species_A"]
    name_B = inputs["name"]["species_B"]

    fig, (ax_A, ax_B) = figax

    def add_data_label(ax, x, y, z):
        prop_dict = {
            'color': 'cyan',
            'fontsize': 6,
        }
        for i in range(len(x)):
            for j in range(len(y)):
                text = f"{z[i][j]:.2f}"
                ax.text(x[i][j], y[i][j], 0.0, text, **prop_dict)

    ax_A.plot_surface(pA_grid, pB_grid, vA_grid, cmap="viridis")
    ax_A.set_xlabel(f"$p_{{{name_A}}} (mTorr)$")
    ax_A.set_ylabel(f"$p_{{{name_B}}} (mTorr)$")
    ax_A.set_zlabel(f"{name_A} layer_thickness")
    ax_A.set_title(f"{name_A}")
    if datalabel:
        add_data_label(ax_A, pA_grid, pB_grid, vA_grid)

    ax_B.plot_surface(pA_grid, pB_grid, vB_grid, cmap="viridis")
    ax_B.set_xlabel(f"$p_{{{name_A}}} (mTorr)$")
    ax_B.set_ylabel(f"$p_{{{name_B}}} (mTorr)$")
    ax_B.set_zlabel(f"{name_B} layer_thickness")
    ax_B.set_title(f"{name_B}")
    if datalabel:
        add_data_label(ax_B, pA_grid, pB_grid, vB_grid)

    fig.suptitle(f"Temperature = {T} K")
    fig.savefig(f"output_{T}.png")
    print(f"Saved output_{T}.png")
    ax_A.cla()
    ax_B.cla()


def write_summary(result, **inputs):
    eV_to_J = inputs["constants"]["eV_to_J"]
    T_list = result["T_list"]
    T = T_list[0]
    E_dict = result["E_dict"]

    Energy = {
        'E_1A': E_dict['E_1A'][T]/eV_to_J,
        'E_1B': E_dict['E_1B'][T]/eV_to_J,
        'E_LA_A': E_dict['E_LA_A'][T]/eV_to_J,
        'E_LA_B': E_dict['E_LA_B'][T]/eV_to_J,
        'E_LB_A': E_dict['E_LB_A'][T]/eV_to_J,
        'E_LB_B': E_dict['E_LB_B'][T]/eV_to_J,
    }

    with open('summary.yaml', 'w') as f:
        yaml.dump(Energy, f)


def search_optimal_condition(dat):
    pA_grid = dat["pA_grid"]
    pB_grid = dat["pB_grid"]
    vA_grid = dat["vA_grid"]
    vB_grid = dat["vB_grid"]

    is_p
    if np.max(pA_grid) < 1e-04:



def main():
    with open('input.yaml') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    T_min = inputs["process_variables"]["temperature"]["min"]
    T_max = inputs["process_variables"]["temperature"]["max"]
    T_step = inputs["process_variables"]["temperature"]["step"]
    T_list = np.arange(T_min, T_max, T_step)

    E_dict = load_energy(T_list, **inputs)
    p_sat_dict = get_saturated_pressure(T_list, E_dict, **inputs)

    result = {
        'T_list': T_list,
        'E_dict': E_dict,
        'p_sat_dict': p_sat_dict,
    }

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6), subplot_kw={"projection": "3d"})
    for T in T_list:
        pA_grid, pB_grid, vA_grid, vB_grid = get_layer_thickness(
            T, E_dict, p_sat_dict, **inputs)
        dat = {
            'pA_grid': pA_grid,
            'pB_grid': pB_grid,
            'vA_grid': vA_grid,
            'vB_grid': vB_grid,
            'T': T,
        }
        plot(dat, figax=(fig, axes), datalabel=True, **inputs)
        search_optimal_condition(dat)

    write_summary(result, **inputs)


if __name__ == "__main__":
    main()
