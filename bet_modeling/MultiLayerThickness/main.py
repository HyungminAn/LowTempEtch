from dataclasses import dataclass
from itertools import product
import numpy as np
import yaml
import matplotlib.pyplot as plt

@dataclass
class CONSTANTS:
    v0 = 1.0E+12  # prefactor in rate constant (s^-1)
    kB = 8.617E-5  # Boltzmann constant (eV/K)
    amu_to_kg = 1.66053906660E-27  # atomic mass unit to kg
    eV_to_J = 1.602176634E-19  # electron volt to Joule
    Pa_to_mTorr = 7.50062  # Pascal to miliTorr

class DataLoader:
    def run(self, T_list, **inputs):
        type_list = [
            ("species_A", "1A"),
            ("species_A", "LA_A"),
            ("species_A", "LA_B"),
            ("species_B", "1B"),
            ("species_B", "LB_A"),
            ("species_B", "LB_B"),
        ]
        result = {}
        for (species, label) in type_list:
            path_E = inputs["path_ads_E"][species][label]
            E_dict = self.read_file(path_E)
            E_dict = {T: E_dict[T] * CONSTANTS.eV_to_J for T in T_list}
            key = f"E_{label}"
            result[key] = E_dict
        return result

    def read_file(self, file_path, type_key=int, type_value=float):
        with open(file_path) as f:
            lines = f.readlines()
        result = {}
        for i in lines:
            key, value = i.strip().split()
            key = type_key(key)
            value = type_value(value)
            result[key] = value
        return result

class SaturationPressureCalculator:
    def run(self, T_list, E_dict, **inputs):
        mass_dict = self.read_file(inputs["path_mass"], type_key=str)
        type_list = [
            ("species_A", "1A"),
            ("species_A", "LA_A"),
            ("species_A", "LA_B"),
            ("species_B", "1B"),
            ("species_B", "LB_A"),
            ("species_B", "LB_B"),
        ]
        result = {}
        for (species, label) in type_list:
            name = inputs["name"][species]
            area = inputs["area"][species]
            mass = mass_dict[name] * CONSTANTS.amu_to_kg
            k_0 = {T: area / (CONSTANTS.v0 * np.sqrt(2*np.pi*mass*CONSTANTS.kB*CONSTANTS.eV_to_J*T)) for T in T_list}
            p_sat = {T: self.get_p_sat(k_0, E_dict[f"E_{label}"], T) for T in T_list}
            result[f'p_{label}_sat'] = p_sat
        return result

    @staticmethod
    def get_p_sat(K, E, T):
        return 1 / (K[T] * np.exp(E[T] / (CONSTANTS.kB * CONSTANTS.eV_to_J * T))) * CONSTANTS.Pa_to_mTorr

    def read_file(self, file_path, type_key=int, type_value=float):
        with open(file_path) as f:
            lines = f.readlines()
        result = {}
        for i in lines:
            key, value = i.strip().split()
            key = type_key(key)
            value = type_value(value)
            result[key] = value
        return result

class LayerThicknessCalculator:
    def run(self, T, p_sat_dict, **inputs):
        '''
        Get the equilibrium layer thickness of species A and B,
        at given temperature T and pressure pA, pB
        '''
        p_1A_sat = p_sat_dict["p_1A_sat"][T]
        p_1B_sat = p_sat_dict["p_1B_sat"][T]
        p_AA_sat = p_sat_dict["p_LA_A_sat"][T]
        p_AB_sat = p_sat_dict["p_LA_B_sat"][T]
        p_BA_sat = p_sat_dict["p_LB_A_sat"][T]
        p_BB_sat = p_sat_dict["p_LB_B_sat"][T]

        pA_grid, pB_grid, vA_grid, vB_grid = self.get_grid(
            p_AA_sat, p_AB_sat, p_BA_sat, p_BB_sat, **inputs)

        n_row, n_col = pA_grid.shape
        for i, j in product(range(n_row), range(n_col)):
            pA, pB = pA_grid[i, j], pB_grid[i, j]
            P0, P = self.get_matrix(
                pA,
                pB,
                p_1A_sat,
                p_1B_sat,
                p_AA_sat,
                p_AB_sat,
                p_BA_sat,
                p_BB_sat)

            if not self.check_convergence(P, pA, pB):
                vA_grid[i, j] = np.nan
                vB_grid[i, j] = np.nan
                continue

            vA, vB = self.calculate_thickness(P, P0)

            vA_grid[i, j] = vA
            vB_grid[i, j] = vB

        return pA_grid, pB_grid, vA_grid, vB_grid

    def get_grid(self, p_AA_sat, p_AB_sat, p_BA_sat, p_BB_sat, **inputs):
        pA_max = min(p_AA_sat, p_AB_sat)
        pB_max = min(p_BA_sat, p_BB_sat)
        pA_ngrid = inputs["pressure"]["species_A"]["n_grid"]
        pB_ngrid = inputs["pressure"]["species_B"]["n_grid"]
        pA_list = np.linspace(0, pA_max, pA_ngrid)
        pB_list = np.linspace(0, pB_max, pB_ngrid)

        pA_grid, pB_grid = np.meshgrid(pA_list, pB_list)
        vA_grid, vB_grid = np.zeros_like(pA_grid), np.zeros_like(pB_grid)
        return pA_grid, pB_grid, vA_grid, vB_grid

    def get_matrix(self, pA, pB,
                   p_1A_sat,
                   p_1B_sat,
                   p_AA_sat,
                   p_AB_sat,
                   p_BA_sat,
                   p_BB_sat):
        ratio_pA_p1A = pA / p_1A_sat
        ratio_pB_p1B = pB / p_1B_sat

        P0_11, P0_12 = ratio_pA_p1A, 0.0
        P0_21, P0_22 = 0.0, ratio_pB_p1B
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
        return P0, P

    def check_convergence(self, P, pA, pB):
        eigenvalues, _ = np.linalg.eig(P)
        magnitudes = np.abs(eigenvalues)
        if np.any(magnitudes >= 1):
            print("Warning: Unstable equilibrium")
            print(f"pA = {pA:.2e}, pB = {pB:.2e}")
            print(f"eigenvalues = {eigenvalues}")
            print(f"magnitudes = {magnitudes}")
            print("#"*79)
            return False
        return True

    def calculate_thickness(self, P, P0):
        v_init = np.array([1.0, 1.0])
        Id_mat = np.eye(2)

        IsubP = Id_mat - P
        inv = np.linalg.inv(IsubP)
        AB = inv @ P0 @ v_init
        AB_tilde = inv @ AB

        A, B = AB
        A_tilde, B_tilde = AB_tilde

        vA = A_tilde / (1 + A + B)
        vB = B_tilde / (1 + A + B)
        return vA, vB

def write_summary(result, **inputs):
    T_list = result["T_list"]
    T = T_list[0]
    E_dict = result["E_dict"]

    Energy = {
        'E_1A': E_dict['E_1A'][T]/CONSTANTS.eV_to_J,
        'E_1B': E_dict['E_1B'][T]/CONSTANTS.eV_to_J,
        'E_LA_A': E_dict['E_LA_A'][T]/CONSTANTS.eV_to_J,
        'E_LA_B': E_dict['E_LA_B'][T]/CONSTANTS.eV_to_J,
        'E_LB_A': E_dict['E_LB_A'][T]/CONSTANTS.eV_to_J,
        'E_LB_B': E_dict['E_LB_B'][T]/CONSTANTS.eV_to_J,
    }

    with open('summary.yaml', 'w') as f:
        yaml.dump(Energy, f)

class ThicknessPlotter:
    def run(self, data, **inputs):
        plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 7))

        ltc = LayerThicknessCalculator()
        p_sat_dict = data['p_sat_dict']
        for T in data['T_list']:
            pA_grid, pB_grid, vA_grid, vB_grid = ltc.run(T, p_sat_dict, **inputs)
            dat = {
                'pA_grid': pA_grid,
                'pB_grid': pB_grid,
                'vA_grid': vA_grid,
                'vB_grid': vB_grid,
                'T': T,
            }
            self.plot(dat, figax=(fig, axes), datalabel=True, **inputs)

    def plot(self, dat, figax=(None, None), **inputs):
        T = dat["T"]
        pA_grid = dat["pA_grid"]
        pB_grid = dat["pB_grid"]
        vA_grid = dat["vA_grid"]
        vB_grid = dat["vB_grid"]

        name_A = inputs["name"]["species_A"]
        name_B = inputs["name"]["species_B"]

        fig, (ax_A, ax_B) = figax

        # --- Plot for Species A (Left Subplot) ---
        # Contour plot for species A layer thickness (vA_grid)
        # Use contourf for filled color regions
        # Use a consistent color mapping (e.g., 'viridis') with specified levels
        levels = np.arange(0, 10.1, 0.5)
        contour_A = ax_A.contourf(pA_grid, pB_grid, vA_grid, levels=levels, cmap="viridis")
        # Add contour lines for better definition
        ax_A.contour(pA_grid, pB_grid, vA_grid, levels=contour_A.levels, colors='k', linewidths=0.5)

        # Add contour lines for better definition (contour_lines_A)
        contour_lines_A = ax_A.contour(pA_grid, pB_grid, vA_grid, levels=contour_A.levels, colors='k', linewidths=0.5)
        ax_A.clabel(contour_lines_A, inline=True, fontsize=10, fmt='%.1f', colors='white')

        # Set labels and title for A
        ax_A.set_xlabel(f"$p_{{\\text{{{name_A}}}}}$ (mTorr)")
        ax_A.set_ylabel(f"$p_{{\\text{{{name_B}}}}}$ (mTorr)")

        # --- Plot for Species B (Right Subplot) ---
        # Contour plot for species B layer thickness (vB_grid)
        contour_B = ax_B.contourf(pA_grid, pB_grid, vB_grid, levels=levels, cmap="plasma") # Changed cmap for differentiation
        # Add contour lines
        ax_B.contour(pA_grid, pB_grid, vB_grid, levels=contour_B.levels, colors='k', linewidths=0.5)

        contour_lines_B = ax_B.contour(pA_grid, pB_grid, vB_grid, levels=contour_B.levels, colors='k', linewidths=0.5)
        ax_B.clabel(contour_lines_B, inline=True, fontsize=10, fmt='%.1f', colors='white')

        # Set labels and title for B
        ax_B.set_xlabel(f"$p_{{\\text{{{name_A}}}}}$ (mTorr)")
        ax_B.set_ylabel(f"$p_{{\\text{{{name_B}}}}}$ (mTorr)")

        # Temporary replace
        name_B = name_B.replace('5', '$_5$')

        # Add a color bar to show the thickness scale
        ax_A.set_title(f"{name_A} Thickness")
        ax_B.set_title(f"{name_B} Thickness")
        fig.colorbar(contour_A, ax=ax_A, label=f'{name_A} layer thickness')
        fig.colorbar(contour_B, ax=ax_B, label=f'{name_B} layer thickness')

        # Set the overall title
        fig.suptitle(f"Layer Thickness Contour Plots at $T = {T}$ K")
        # Adjust layout to prevent overlap
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout()

        # Save the figure
        fig.savefig(f"output_{T}.png", dpi=200)
        fig.savefig(f"output_{T}.pdf")
        fig.savefig(f"output_{T}.eps")
        print(f"Saved output_{T}.png")

        # Clear axes for the next iteration in the run method
        ax_A.cla()
        ax_B.cla()

def main():
    with open('input.yaml') as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)

    T_min = inputs["temperature"]["min"]
    T_max = inputs["temperature"]["max"]
    T_step = inputs["temperature"]["step"]
    T_list = np.arange(T_min, T_max, T_step)

    dl = DataLoader()
    E_dict = dl.run(T_list, **inputs)

    spc = SaturationPressureCalculator()
    p_sat_dict = spc.run(T_list, E_dict, **inputs)

    data = {
        'T_list': T_list,
        'E_dict': E_dict,
        'p_sat_dict': p_sat_dict,
    }

    tp = ThicknessPlotter()
    tp.run(data, **inputs)

    write_summary(data, **inputs)


if __name__ == "__main__":
    main()
