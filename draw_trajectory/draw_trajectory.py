import sys
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from collections import defaultdict
from itertools import cycle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ase.io import read
from ase.geometry import get_distances
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
                atom_in_cluster_idx = atom_in_cluster_idx.flatten()
                mol_dict[AtomInfo.molecule_dict[tuple(formula)]].append(atom_in_cluster_idx)
            else:
                line = f'Unassigned molecule with formula {formula}\n'
                line += f' at atoms {atom_in_cluster_idx}\n'
                line += f' with positions {ase_image.get_positions()}\n'

                raise UnAssignedError(line)

        return mol_dict


class UnAssignedError(Exception):
    pass


def plot(cell, positions):
    color_list = ['#0c74b2',
                  '#D76224',
                  '#1BA077',
                  '#CA7AAA',
                  '#E8A125']
    color_cycle = cycle(color_list)
    x_lim, y_lim = cell[0, 0], cell[1, 1]
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(8.5*cm, 8.5*cm))

    positions[:, :, :] -= positions[0, :, :]
    x_min = np.min(positions[:, :, 0])
    x_max = np.max(positions[:, :, 0])
    y_min = np.min(positions[:, :, 1])
    y_max = np.max(positions[:, :, 1])
    x_min_int = int(np.floor(x_min / x_lim))
    x_max_int = int(np.ceil(x_max / x_lim))
    y_min_int = int(np.floor(y_min / y_lim))
    y_max_int = int(np.ceil(y_max / y_lim))

    x_min = x_min_int * x_lim
    x_max = x_max_int * x_lim
    y_min = y_min_int * y_lim
    y_max = y_max_int * y_lim
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for atom_idx, color in zip(range(positions.shape[1]), color_cycle):
        x, y = positions[:, atom_idx, 0], positions[:, atom_idx, 1]

        line_prop_dict = {
                'color': color,
                'linewidth': 1,
                'alpha': 0.5,
                }
        ax.plot(x, y, **line_prop_dict)

    axline_prop_dict = {
            'color': 'grey',
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 0.5,
            }
    for i in range(x_min_int + 1, x_max_int):
        ax.axvline(i * x_lim, **axline_prop_dict)
    for i in range(y_min_int + 1, y_max_int):
        ax.axhline(i * y_lim, **axline_prop_dict)
    ax.set_aspect(y_lim / x_lim)
    ax.set_xlabel('Coordinate(Å)')
    ax.set_ylabel('Coordinate(Å)')

    circle_prop_dict = {
            'edgecolor': 'black',
            'facecolor': None,
            'linestyle': '--',
            'fill': False,
            'alpha': 0.5,
            }
    circle = matplotlib.patches.Circle((0, 0), 3, **circle_prop_dict)
    ax.add_patch(circle)
    fig.tight_layout()
    fig.savefig('trajectory.png')


def get_unwrapped_positions(dump, method):
    positions = [atoms.get_positions() for atoms in dump]
    positions = np.array(positions)

    if method == 'dump':
        return positions

    elif method == 'xdatcar':
        unwrapped_positions = [positions[0, :, :].squeeze()]
        cell = dump[0].get_cell()
        for image_idx in range(1, len(positions)):
            D, _ = get_distances(positions[image_idx-1, :, :].squeeze(),
                                 positions[image_idx, :, :].squeeze(),
                                 cell=cell,
                                 pbc=True)
            diff = np.diagonal(D, axis1=0, axis2=1).transpose()
            unwrapped_positions.append(unwrapped_positions[-1] + diff)

            if image_idx % 100 == 0:
                print(f'Unwrapping {image_idx}th image')
        unwrapped_positions = np.array(unwrapped_positions)
        return unwrapped_positions

    else:
        raise ValueError(f'Invalid method {method} (possible options: dump/xdatcar)')


def filter_positions(dump, cutoff):
    positions = [atoms.get_positions() for atoms in dump]
    max_h = np.zeros(len(positions[0]))
    idx_to_filter = []
    for pos in positions:
        max_h = np.maximum(max_h, pos[:, 2])
    for i, h in enumerate(max_h):
        if h > cutoff:
            idx_to_filter.append(i)
    return idx_to_filter


def main():
    if len(sys.argv) != 3:
        print('Usage: python draw_trajectory.py <dump/XDATCAR> dump/xdatcar')
        sys.exit(1)

    path_to_dump = sys.argv[1]
    method = sys.argv[2]

    dump = read(path_to_dump, index=':')
    poscar_init = dump[0]
    mol_dict = MoleculeInfoAllocator.run(poscar_init)
    cutoff = poscar_init.get_cell().diagonal().max() - 5.0
    idx_to_filter = filter_positions(dump, cutoff)
    track_idx = [i if poscar_init[i].symbol == 'F' else j
                 for (i, j) in mol_dict['HF']]
    track_idx = np.array([i for i in track_idx if i not in idx_to_filter])

    positions = get_unwrapped_positions(dump, method)
    my_positions = positions[:, track_idx, :]
    cell = poscar_init.get_cell()
    plot(cell, my_positions)


if __name__ == '__main__':
    main()
