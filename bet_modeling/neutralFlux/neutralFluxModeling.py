from math import sqrt
from math import exp

import numpy as np
import matplotlib.pyplot as plt

import yaml


def knudsen_flux(params, to_print=False):
    '''
    Calculate the Knudsen flux of a neutral gas through a hole.
    '''
    constants = params['constants']
    process_params = params['process_params']
    gas_params = params['gas_params']

    P = process_params['P']
    L = process_params['L']
    T = process_params['T']
    d = process_params['d']

    pi = constants['pi']
    kB = constants['kB']
    # N_a = constants['N_a']

    m = gas_params['m']

    v_1 = (P/L)
    v_2 = (pi * d**3)
    # v_3 = 1 / (12*kB*N_a*T)
    v_3 = 1 / (12*kB*T)
    v_4 = sqrt(8*kB*T/(pi*m))

    J = v_1 * v_2 * v_3 * v_4

    if to_print:
        print('*'*80)
        print('Knudsen flux', end='\n\t')
        print(f'P/L: {v_1:.2e}', end='\n\t')
        print(f'pi*d^3: {v_2:.2e}', end='\n\t')
        # print(f'1/(12*kB*N_a*T): {v_3:.2e}', end='\n\t')
        print(f'1/(12*kB*T): {v_3:.2e}', end='\n\t')
        print(f'sqrt(8*kB*T/(pi*m)): {v_4:.2e}', end='\n\n\t')
        print(f'J_K = {v_1:.2e} * {v_2:.2e} * {v_3:.2e} * {v_4:.2e} = {J:.2e}')
    return J


def surface_flux(params, to_print=False):
    '''
    Calculate the surface flux of a neutral gas through a hole.
    '''
    constants = params['constants']
    process_params = params['process_params']
    gas_params = params['gas_params']

    P = process_params['P']
    L = process_params['L']
    T = process_params['T']
    d = process_params['d']

    pi = constants['pi']
    kB = constants['kB']
    k0 = constants['k0']

    m = gas_params['m']
    D0 = gas_params['D0']
    E_ads = gas_params['E_ads']
    E_diff = gas_params['E_diff']

    v_1 = (P/L)
    v_2 = sqrt(pi)*d*D0
    v_3 = 1 / (k0 * sqrt(2*kB*m*T))
    v_4 = exp((E_ads-E_diff)/(kB*T))

    J = v_1 * v_2 * v_3 * v_4

    if to_print:
        print('*'*80)
        print('Surface flux', end='\n\t')
        print(f'P/L: {v_1:.2e}', end='\n\t')
        print(f'sqrt(pi)*d*D0: {v_2:.2e}', end='\n\t')
        print(f'1/(k0*sqrt(2*kB*m*T)): {v_3:.2e}', end='\n\t')
        print(f'exp((E_ads-E_diff)/(kB*T)): {v_4:.2e}', end='\n\n\t')
        print(f'J_S = {v_1:.2e} * {v_2:.2e} * {v_3:.2e} * {v_4:.2e} = {J:.2e}')

    return J


def print_params(params):
    '''
    Print the parameters in a dictionary.
    '''
    print('*'*80)
    for name, params_dict in params.items():
        print(name + ':')
        for key, value in params_dict.items():
            if isinstance(value, float):
                print(f'\t{key}: {value:.2e}')
            else:
                print(f'\t{key}: {value}')


def flux_ratio(params, to_print=False):
    '''
    Get the ratio of the Surface flux to the Knudsen flux.
    '''
    constants = params['constants']
    process_params = params['process_params']
    gas_params = params['gas_params']

    kB = constants['kB']
    # N_a = constants['N_a']
    D0 = gas_params['D0']
    k0 = constants['k0']
    E_ads = gas_params['E_ads']
    E_diff = gas_params['E_diff']
    T = process_params['T']
    d = process_params['d']

    # v_1 = 3*N_a*D0/(k0*d**2)
    v_1 = 3*D0/(k0*d**2)
    v_2 = exp((E_ads-E_diff)/(kB*T))

    if to_print:
        print('*'*80)
        print('Flux ratio', end='\n\t')
        # print(f'3*N_a*D0/(k0*d^2): {v_1:.2e}', end='\n\t')
        print(f'3*D0/(k0*d^2): {v_1:.2e}', end='\n\t')
        print(f'exp((E_ads-E_diff)/(kB*T)): {v_2:.2e}', end='\n\t')
        print(f'Flux ratio: {v_1*v_2:.2e}')


def convert_units(params):
    '''
    Convert the units of the parameters to the correct units.
    '''
    constants = params['constants']
    process_params = params['process_params']
    gas_params = params['gas_params']

    process_params['L'] = process_params['d'] * process_params['AR']

    eV_to_J = 1.602*10**-19  # conversion factor from eV to J
    amu_to_kg = 1.661*10**-27  # conversion factor from amu to kg
    cm2_to_m2 = 1E-04  # conversion factor from cm^2 to m^2

    constants['kB'] *= eV_to_J  # convert Boltzmann constant to J/K
    gas_params['m'] *= amu_to_kg  # convert mass to kg
    gas_params['E_ads'] *= eV_to_J  # convert adsorption energy to J
    gas_params['E_diff'] *= eV_to_J  # convert diffusion energy to J
    gas_params['D0'] *= cm2_to_m2  # convert diffusion coefficient to m^2/s


def single_calculation(params, to_print=False):
    '''
    Perform a single calculation of the total flux.
    '''
    J_knudsen = knudsen_flux(params, to_print=to_print)
    J_surf = surface_flux(params, to_print=to_print)
    J_total = J_knudsen + J_surf
    flux_ratio(params, to_print=to_print)

    if to_print:
        print('*'*80)
        print('Knudsen flux: J_K = {:.2e} /s'.format(J_knudsen))
        print('Surface flux: J_S = {:.2e} /s'.format(J_surf))
        print('J_S/J_K = {:.2e}'.format(J_surf/J_knudsen))
        print('Total flux: {:.2e} /s'.format(J_total))


def get_data(params, var_info, func):
    v_min, v_max, n_data = var_info['var_range']
    var_name = var_info['var_name']
    var_type = var_info['type']

    if var_info.get('log_xscale'):
        x = np.logspace(np.log10(v_min), np.log10(v_max), n_data)
    else:
        x = np.linspace(v_min, v_max, n_data)

    y = []
    for i in x:
        params[var_type][var_name] = i
        value = func(params)
        y.append(value)
    y = np.array(y)
    return x, y


def plot_along_variable(params, var_info):
    '''
    Plot the total flux as a function of a variable.
    '''
    x, J_K = get_data(params, var_info, knudsen_flux)
    x, J_S = get_data(params, var_info, surface_flux)

    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    ax, ax_ratio = axes

    ax.plot(x, J_K, label='Knudsen flux')
    ax.plot(x, J_S, label='Surface flux')
    ax.legend()
    if var_info.get('log_xscale'):
        ax.set_xscale('log')
        ax.set_xlabel(f'log ({var_info["var_name"]})')
    else:
        ax.set_xlabel(var_info['var_name'])

    if var_info.get('log_yscale'):
        ax.set_ylabel('log(Flux/s)')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Flux/s')

    ax_ratio.plot(x, J_S/J_K, label='Surface/Knudsen', color='black')
    if var_info.get('log_xscale'):
        ax_ratio.set_xscale('log')
        ax_ratio.set_xlabel(f'log ({var_info["var_name"]})')
    else:
        ax_ratio.set_xlabel(var_info['var_name'])

    if var_info.get('log_yscale'):
        ax_ratio.set_ylabel('log(Flux ratio)')
        ax_ratio.set_yscale('log')
    else:
        ax_ratio.set_ylabel('Flux ratio')

    ax_ratio.axhline(1, color='grey', linestyle='--')
    # plot the x value that is close to the flux ratio of 1
    idx = np.argmin(np.abs(J_S/J_K - 1))
    ax_ratio.axvline(
        x[idx], color='red', linestyle='--',
        label=f'{var_info["var_name"]} = {x[idx]:.2e}')
    ax_ratio.legend()

    fig.tight_layout()
    process_type = var_info['type']
    var_name = var_info['var_name']
    fig.savefig(f'flux_vs_{process_type}_{var_name}.png')


def main():
    to_print = False

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load data
    convert_units(params)
    if to_print:
        print_params(params)
    # single_calculation(params, to_print=to_print)

    # var_info = {
    #     'type': 'process_params',
    #     'var_name': 'T',
    #     'var_range': [100, 1000, 50],
    # }

    var_info = {
        'type': 'process_params',
        'var_name': 'd',
        'var_range': [1E-08, 1E-05, 100],
        'log_xscale': True,
        'log_yscale': True,
    }

    plot_along_variable(params, var_info)


if __name__ == '__main__':
    main()
