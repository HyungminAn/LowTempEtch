import sys
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from collections import defaultdict
from itertools import cycle
from pprint import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from graph_tool import Graph
from graph_tool.topology import label_components


@dataclass
class AtomInfo:
    bond_length = {
            ('I', 'F'): 1.85393 * 1.3,
            ('H', 'F'): 0.937914 * 1.3,
            ('N', 'H'): 1.02728 * 1.3,
            ('Si', 'F'): 1.72613 * 1.3,
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
        cluster_idx.pop(slab_idx)

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


def is_independent_HF(mol_type_i, mol_type_j):
    if (
            set([mol_type_i, mol_type_j]) == set(['NH4', 'SiF6'])
            or set([mol_type_i, mol_type_j]) == set(['NH4', 'IF5'])
            or set([mol_type_i, mol_type_j]) == set(['NH4', 'NH4'])
            or set([mol_type_i, mol_type_j]) == set(['SiF6', 'IF5'])
            or set([mol_type_i, mol_type_j]) == set(['SiF6', 'SiF6'])
            or set([mol_type_i, mol_type_j]) == set(['IF5', 'IF5'])
            ):
        # print(f'This molecule is an independent single molecule')
        return True
    else:
        return False

def is_start_node(mol_type_i, mol_type_j):
    if (
            set([mol_type_i, mol_type_j]) == set(['NH4', 'HF'])
            or set([mol_type_i, mol_type_j]) == set(['SiF6', 'HF'])
            or set([mol_type_i, mol_type_j]) == set(['IF5', 'HF'])
            ):
        # print(f'This molecule is a start node')
        return True
    else:
        return False


def is_terminated_HF(i, j, idx_nn_i, idx_nn_j, image, dist):
    symbols = image.get_chemical_symbols()
    symbol_i = symbols[i]
    symbol_j = symbols[j]
    symbol_nn_i = symbols[idx_nn_i]
    symbol_nn_j = symbols[idx_nn_j]

    key_i = frozenset([i, idx_nn_i])


def analyze_hf_chain(image, mol_dict):
    molecule_mapping = {}
    for mol_type, idxes in mol_dict.items():
        for idx in idxes:
            molecule_mapping |= {i: mol_type for i in idx}
    cluster_mapping = {}
    for idxes in mol_dict.values():
        for idx in idxes:
            for i in idx:
                cluster_mapping[i] = idx

    dist = image.get_all_distances(mic=True)
    np.fill_diagonal(dist, np.inf)

    dict_single_HF = {}
    dict_node_HF = {}
    dict_chain_HF = {}
    dict_terminated_HF = {}

    for i, j in mol_dict['HF']:
        idx_nn_i = np.argpartition(dist[i], 1)[1]
        idx_nn_j = np.argpartition(dist[j], 1)[1]
        mol_type_i = molecule_mapping[idx_nn_i]
        mol_type_j = molecule_mapping[idx_nn_j]
        # pp(f'HF (idx {i}, {j}) is connected to {mol_type_i} (idx {idx_nn_i}) and {mol_type_j} (idx {idx_nn_j})')

        key = frozenset([i, j])
        value = [idx_nn_i, idx_nn_j]
        if is_terminated_HF(i, j, idx_nn_i, idx_nn_j, image, dist):
            dict_terminated_HF[key] = value
        elif is_independent_HF(mol_type_i, mol_type_j):
            dict_single_HF[key] = value
        elif is_start_node(mol_type_i, mol_type_j):
            dict_node_HF[key] = value
        else:
            dict_chain_HF[key] = value

    result = {}
    done_dict = set()
    for ij, (idx_nn_i, idx_nn_j) in dict_node_HF.items():
        if ij in done_dict:
            continue
        done_dict.add(ij)

        chain_HF = [ij]
        next_chain_idx = idx_nn_i if molecule_mapping[idx_nn_i] == 'HF' else idx_nn_j
        search_key = cluster_mapping[next_chain_idx]
        if search_key in dict_node_HF:
            chain_HF.append(search_key)
            done_dict.add(search_key)

            chain_no = len(result)
            result[chain_no] = chain_HF
            continue

        while True:
            chain_HF.append(search_key)
            done_dict.add(search_key)
            pp(f'Current chain: {chain_HF}, search_key: {search_key}')
            breakpoint()

            (idx_nn_i, idx_nn_j) = dict_chain_HF[search_key]
            if cluster_mapping[idx_nn_i] not in done_dict:
                search_key = cluster_mapping[idx_nn_i]
            elif cluster_mapping[idx_nn_j] not in done_dict:
                search_key = cluster_mapping[idx_nn_j]
            else:
                assert cluster_mapping[idx_nn_i] in done_dict or cluster_mapping[idx_nn_j] in done_dict

            if search_key in dict_node_HF or search_key in dict_terminated_HF:
                chain_HF.append(search_key)
                done_dict.add(search_key)

                chain_no = len(result)
                result[chain_no] = chain_HF
                break

    output = {
            'single_HF': dict_single_HF,
            'node_HF': dict_node_HF,
            'chain_HF': dict_chain_HF,
            'terminated_HF': dict_terminated_HF,
            'chain': result,
            }
    return output


def check_chain_length(output):
    total_length = []
    for chain in output['chain'].values():
        total_length.append(len(chain))
    for chain in output['terminated_HF'].values():
        total_length.append(1)
    for chain in output['single_HF'].values():
        total_length.append(1)
    return sum(total_length) / len(total_length)


def main():
    if len(sys.argv) != 2:
        print('Usage: python chain_analysis_HF.py <path_to_dump>')
        sys.exit(1)
    path_dump = sys.argv[1]
    dump = read(path_dump, index=':')
    positions = [atoms.get_positions() for atoms in dump]
    positions = np.array(positions)

    chain_lengths = []
    # for image in dump:
    for image in [dump[2]]:
        mol_dict = MoleculeInfoAllocator.run(image)
        output = analyze_hf_chain(image, mol_dict)
        chain_length = check_chain_length(output)
        chain_lengths.append(chain_length)
        pp(output)
        print(f'Chain length: {chain_length}')


if __name__ == '__main__':
    main()
