import matplotlib.pyplot as plt
import numpy as np


def main():
    mass = 20.01  # amu unit
    amu_to_kg = 1.66e-27  # kg unit

    kB = 1.38e-23  # J/K unit
    v0 = 1E+12  # Hz unit
    eV_to_J = 1.6e-19  # J unit
    Area = 5.8774  # A^2 unit
    A2_to_m2 = 1E-20
    pressure = 1.0  # Pa unit

    T_min, T_max, T_step = 200, 300, 1
    E_min, E_max, E_step = 0.01, 0.6, 0.01
    T_list = np.arange(T_min, T_max, T_step)
    E_list = np.arange(E_min, E_max, E_step)
    T_grid, E_grid = np.meshgrid(T_list, E_list)

    k_ads = pressure * Area * A2_to_m2 / np.sqrt(2 * np.pi * mass * amu_to_kg * kB * T_grid)
    k_des = v0 * np.exp(-E_grid * eV_to_J / (kB * T_grid))
    Kp = k_ads / k_des
    theta = Kp / (1 + Kp)

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))
    ax.plot_surface(T_grid, E_grid, theta, cmap='viridis')

    row, col = T_grid.shape
    for i in range(0, row, 10):
        for j in range(0, col, 10):
            ax.text(T_grid[i][j], E_grid[i][j], 0.0, f"{theta[i][j]:.2f}",
                    color='deepskyblue')

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("E_ads (eV)")
    ax.set_zlabel("Theta (coverage)")
    ax.set_title("Isotherm of adsorption HF")

    fig.tight_layout()
    fig.savefig("result.png")


if __name__ == "__main__":
    main()
