import numpy as np

from ase.units import bar
from ase.io import write


class RelaxLogger():
    def __init__(self):
        self.logfile = None

    def _set_logfile(self, logfile):
        self.logfile = logfile

    def _log_initial_structure(self, atoms):
        '''
        Log the initial structure to the *logfile*
        '''
        if self.logfile is None:
            raise ParameterUnsetError('logfile')

        w = self.logfile.write
        _cell = atoms.get_cell()
        w("Cell\n")
        for ilat in range(0, 3):
            line = f"{_cell[ilat][0]:.6f}  "
            line += f"{_cell[ilat][1]:.6f}  "
            line += f"{_cell[ilat][2]:.6f}\n"
            w(line)

        _pbc = atoms.get_pbc()
        w("PBCs\n")
        w(f"{_pbc[0]}  {_pbc[1]}  {_pbc[2]}\n")
        _pos = atoms.get_positions()
        _mass = atoms.get_masses()
        _chem_sym = atoms.get_chemical_symbols()
        w("Atoms\n")
        for i in range(len(_chem_sym)):
            line = f"{_chem_sym[i]}  "
            line += f"{_mass[i]:.6f}  "
            line += f"{_pos[i][0]:.6f}  "
            line += f"{_pos[i][1]:.6f}  "
            line += f"{_pos[i][2]:.6f}\n"
            w(line)

    def _initialize_relax_log(self, traj, thermo):
        self.traj = traj
        self.thermo = thermo

        with open(traj, 'w') as _:
            pass

        # Define a function to be called at each optimization step
        with open(thermo, 'w') as f:
            line = f"{'step':>8s}  {'PotEng':>15s}  {'volume':>15s}  "
            line += f"{'press':>15s}  "
            line += f"{'p_xx':>15s}  {'p_yy':>15s}  {'p_zz':>15s}  "
            line += f"{'p_xy':>15s}  {'p_yz':>15s}  {'p_zx':>15s}\n"
            f.write(line)

    def _save_info(self, atoms, step):
        """
        Save atomic positions to a trajectory file
            and the stress tensor to another file.
        Voigt order: xx, yy, zz, yz, xz, xy
        Saved order: xx, yy, zz, xy, yz, zx
        """
        if self.traj is None:
            raise ParameterUnsetError('traj')
        if self.thermo is None:
            raise ParameterUnsetError('thermo')
        thermo_dat = self.thermo
        extxyz_filename = self.traj

        PotEng = atoms.get_potential_energy()
        volume = atoms.get_volume()
        stress = atoms.get_stress() / (-bar)
        press = np.average(stress[0:3])

        line = f"{step:8d}  {PotEng:15.4f}  {volume:15.4f}  "
        line += f"{press:15.4f}  "
        line += f"{stress[0]:15.4f}  {stress[1]:15.4f}  {stress[2]:15.4f}  "
        line += f"{stress[5]:15.4f}  {stress[3]:15.4f}  {stress[4]:15.4f}\n"

        with open(thermo_dat, 'a') as f:
            f.write(line)

        write(extxyz_filename, atoms, format='extxyz', append=True)


class ParameterUnsetError(Exception):
    def __init__(self, param_name):
        self.param_name = param_name

    def __str__(self):
        return f"Parameter {self.param_name} is not set"
