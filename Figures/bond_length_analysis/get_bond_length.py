import os
import sys
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from collections import defaultdict
import pickle
from pprint import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from graph_tool import Graph
from graph_tool.topology import label_components

@dataclass
class Constants:
    bond_length_crit = 1.25


@dataclass
class AtomInfo:
    bond_length = {
            frozenset(('I', 'F')): 1.85393 * Constants.bond_length_crit,
            frozenset(('H', 'F')): 0.937914 * Constants.bond_length_crit,
            frozenset(('N', 'H')): 1.02728 * Constants.bond_length_crit,
            frozenset(('Si', 'F')): 1.72613 * Constants.bond_length_crit,
            }

    elem_idx = {
            'H': 0,
            'N': 1,
            'F': 2,
            'Si': 3,
            'I': 4,
            }

    molecule_dict = {
            (0, 0, 6, 1, 0): 'SiF6',
            (1, 0, 1, 0, 0): 'HF',
            (4, 1, 0, 0, 0): 'NH4',
            (0, 0, 5, 0, 1): 'IF5',
            }


def generate_bond_mat():
    bond_length_dict = AtomInfo.bond_length
    elem_idx_dict = AtomInfo.elem_idx

    n_elem = len(elem_idx_dict)
    bond_length_mat = np.zeros((n_elem, n_elem))
    for (elem1, elem2), bond_length in bond_length_dict.items():
        idx1, idx2 = elem_idx_dict[elem1], elem_idx_dict[elem2]
        bond_length_mat[idx1, idx2] = bond_length
        bond_length_mat[idx2, idx1] = bond_length
    return bond_length_mat


class AtomImageWithGraph():
    def __init__(self, ase_image):
        self.bond_length = generate_bond_mat()
        self.n_elements = len(self.bond_length)
        self.image = ase_image
        self.num_atoms = len(self.image)
        self.atomic_numbers = np.array([
            AtomInfo.elem_idx[i]
            for i in self.image.get_chemical_symbols()
            ])

        self.find_NN()
        self.draw_graph()

    def find_nearest_neighbors(self, i):
        '''
        Find nearest neighbors for atom i within the cutoff_distance.
        '''
        n_atoms = self.num_atoms
        elem_idx = self.atomic_numbers
        bl_mat = self.bond_length

        indices = np.arange(n_atoms)
        distances = self.image.get_distances(i, indices, mic=True)
        neighbors_logical = np.array([
            distances[j] < bl_mat[elem_idx[i], elem_idx[j]]
            for j in indices
        ])
        neighbors_logical[i] = False
        neighbors = np.where(neighbors_logical)

        return (i, neighbors)

    def find_NN(self):
        '''
        Create a multiprocessing Pool,
            and run the find_nearest_neighbors function for each atom.
        '''
        pool = Pool(cpu_count())
        self.nearest_neighbor = pool.starmap(
            self.find_nearest_neighbors,
            [(i, ) for i in range(self.num_atoms)])

    def draw_graph(self):
        self.graph = Graph(directed=False)
        self.graph.add_vertex(self.num_atoms)

        for (idx, neighbors) in self.nearest_neighbor:
            if neighbors[0].size == 0:
                continue

            for j in neighbors[0]:
                self.graph.add_edge(idx, j)


class MoleculeInfoAllocator():
    @staticmethod
    def run(ase_image):
        mol_dict = defaultdict(list)

        image = AtomImageWithGraph(ase_image)
        atom_idx = image.atomic_numbers
        cluster, hist = label_components(image.graph)
        slab_idx = np.argmax(hist)
        cluster_idx = [i for i in range(len(hist))]

        for i in cluster_idx:
            atom_in_cluster_idx = np.argwhere(cluster.a == i)
            formula = np.zeros(image.n_elements, dtype=int)

            for j in atom_in_cluster_idx:
                formula[atom_idx[j]] += 1

            if tuple(formula) in AtomInfo.molecule_dict:
                atom_in_cluster_idx = frozenset(atom_in_cluster_idx.flatten())
                mol_dict[AtomInfo.molecule_dict[tuple(formula)]].append(atom_in_cluster_idx)
            else:
                line = f'Unassigned molecule with formula {formula}\n'
                line += f' at atoms {atom_in_cluster_idx}\n'
                line += f' with positions {ase_image.get_positions()}\n'

                raise UnAssignedError(line)

        return mol_dict


class UnAssignedError(Exception):
    pass


def check_bondlength(image,
                     mol_match_dict,
                     neighbor_dict,
                     bond_length_dict,
                     center_atom_symbol,
                     mol_type):
    for atom_idx, neighbors in neighbor_dict.items():
        if image[atom_idx].symbol != center_atom_symbol:
            continue

        if mol_match_dict[atom_idx] != mol_type:
            continue

        for neighbor in neighbors:
            assert mol_match_dict[neighbor] == mol_type

            distance = image.get_distance(atom_idx, neighbor, mic=True)
            key = frozenset([atom_idx, neighbor])
            bond_length_dict[key].append(distance)


def get_mol_dict(dump):
    mol_dict = MoleculeInfoAllocator.run(dump[0])
    mol_match_dict = {}
    neighbor_dict = {}
    for mol_type, mols in mol_dict.items():
        for atoms in mols:
            for atom_idx in atoms:
                neighbor_dict[atom_idx] = [i for i in atoms if i != atom_idx]
                mol_match_dict[atom_idx] = mol_type
    return mol_match_dict, neighbor_dict


def generate_bondlength_data(dump, center_atom_symbol, mol_type):
    path_save = f"./bond_length_dict_{mol_type}.pkl"
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            return pickle.load(f)

    mol_match_dict, neighbor_dict = get_mol_dict(dump)
    bond_length_dict = defaultdict(list)
    for idx, image in enumerate(dump):
        check_bondlength(image,
                         mol_match_dict,
                         neighbor_dict,
                         bond_length_dict,
                         center_atom_symbol,
                         mol_type)
        print(f'Processed image {idx}')

    with open(path_save, 'wb') as f:
        pickle.dump(bond_length_dict, f)

    return bond_length_dict


def plot(bond_length_dict, mol_type, logstep):
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(8.5*cm, 8.5*cm))
    line_prop = {
            'alpha': 0.5,
            'linewidth': 0.5,
            'linestyle': '-',
            }
    x = [i * logstep for i in range(len(list(bond_length_dict.values())[0]))]
    for y in bond_length_dict.values():
        ax.plot(x, y, **line_prop)

    ax.set_title(f'{mol_type}')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Bond length ($\AA$)')
    fig.tight_layout()
    fig.savefig(f'bond_length_{mol_type}.png')
    fig.savefig(f'bond_length_{mol_type}.pdf')


def main():
    if len(sys.argv) != 3:
        print('Usage: python chain_analysis_HF.py <path_to_dump> <logstep>')
        sys.exit(1)
    path_dump = sys.argv[1]
    logstep = int(sys.argv[2])
    dump = read(path_dump, index=':')
    mol_type_list = [
            ('Si', 'SiF6'),
            ('H', 'HF'),
            ('N', 'NH4'),
            ('I', 'IF5'),
            ]

    for center_atom_symbol, mol_type in mol_type_list:
        bond_length_dict = generate_bondlength_data(dump,
                                                    center_atom_symbol,
                                                    mol_type)
        plot(bond_length_dict, mol_type, logstep)


if __name__ == '__main__':
    main()
