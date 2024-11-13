import os
import yaml


def main():
    src = "/data2/andynn/LowTempEtch/07_Equil_LayerThickness"

    with open(f"{src}/08_Calculate/mol_list.dat", 'r') as f:
        lines = f.readlines()
        species_list = [line.strip() for line in lines]

    for species in species_list:
        src_1A = f"{src}/03_bareSurface/prev_results/HF/plot/EffectiveAds/dat"
        src_1B = f"{src}/03_bareSurface/results/{species}/plot/EffectiveAds/dat"
        src_LA_A = f"{src}/04_HFonHF/HF/plot/EffectiveAds/dat"
        src_LA_B = f"{src}/06_HFonADDI/results/{species}/plot/EffectiveAds/dat"
        src_LB_A = f"{src}/05_ADDIonHF/results/{species}/plot/EffectiveAds/dat"
        src_LB_B = f"{src}/07_ADDIonADDI/results/{species}/plot/EffectiveAds/dat"

        path_gen_Cell = f"{src}/01_genCell/results/{species}/result.yaml"
        with open(path_gen_Cell, 'r') as f:
            gen_Cell_result = yaml.load(f, Loader=yaml.FullLoader)
            n_mol = gen_Cell_result['params']['n_mol']
            cell_area = 117.609961  # A^2
            area_B = cell_area / n_mol * 1E-20  # m^2

        input_dict = {
            'constants': {
                'v0': 1.0E+12,  # prefactor in rate constant (s^-1)
                'kB': 8.617E-5,  # Boltzmann constant (eV/K)
                'amu_to_kg': 1.66053906660E-27,  # atomic mass unit to kg
                'eV_to_J': 1.602176634E-19,  # electron volt to Joule
                'Pa_to_mTorr': 7.50062,  # Pascal to miliTorr
            },

            'name': {
                'species_A': 'HF',
                'species_B': species,
            },
            'area': {
                'species_A': 5.877425E-20,
                'species_B': area_B,
            },

            'path_mass': f"{src}/08_Calculate/mass.dat",
            'path_ads_E': {
                'species_A': {
                    '1A': src_1A,
                    'LA_A': src_LA_A,
                    'LA_B': src_LA_B,
                },
                'species_B': {
                    '1B': src_1B,
                    'LB_A': src_LB_A,
                    'LB_B': src_LB_B,
                },
            },

            'process_variables': {
                'temperature': {
                    'min': 250,
                    'max': 300,
                    'step': 100,
                },
                'pressure': {
                    'species_A': {
                        'n_grid': 10,
                    },
                    'species_B': {
                        'n_grid': 10,
                    },
                },
            },

        }

        os.makedirs(species, exist_ok=True)
        dst = f"{species}/input.yaml"

        with open(dst, 'w') as f:
            yaml.dump(input_dict, f)

        print(f"Generated {dst}")


if __name__ == '__main__':
    main()
