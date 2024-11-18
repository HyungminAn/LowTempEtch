import os
import subprocess
import shlex
import warnings
from tempfile import mktemp as uns_mktemp
from tempfile import NamedTemporaryFile

import numpy as np

from ase.calculators.lammpsrun import LAMMPS, SpecialTee
from ase.io.lammpsrun import read_lammps_dump
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.lammps import Prism, convert, write_lammps_in


class MyLAMMPS(LAMMPS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, set_atoms=False):
        # !TODO: split this function
        """Method which explicitly runs LAMMPS."""
        self._run_set_info()

        # setup file names for LAMMPS calculation
        filepaths = self._run_set_filepaths()

        # write LAMMPS input and get output
        lmp_handle = self._run_write_input_and_get_output(**filepaths)
        while not self._run_check_success():
            lmp_handle = self._run_write_input_and_get_output(**filepaths)

        # write output
        self._run_write_output(lmp_handle, set_atoms=set_atoms, **filepaths)

    def _run_set_info(self):
        pbc = self.atoms.get_pbc()
        if all(pbc):
            cell = self.atoms.get_cell()
        elif not any(pbc):
            # large enough cell for non-periodic calculation -
            # LAMMPS shrink-wraps automatically via input command
            #       "periodic s s s"
            # below
            cell = 2 * np.max(np.abs(self.atoms.get_positions())) * np.eye(3)
        else:
            warnings.warn(
                "semi-periodic ASE cell detected - translation "
                + "to proper LAMMPS input cell might fail"
            )
            cell = self.atoms.get_cell()
        self.prism = Prism(cell)

        self.set_missing_parameters()
        self.calls += 1

    def _run_set_filepaths(self):
        # change into subdirectory for LAMMPS calculations
        tempdir = self.parameters['tmp_dir']

        # setup file names for LAMMPS calculation
        label = f"{self.label}{self.calls:>06}"
        lammps_in = uns_mktemp(prefix="in_" + label, dir=tempdir)
        lammps_log = uns_mktemp(prefix="log_" + label, dir=tempdir)
        lammps_trj_fd = NamedTemporaryFile(
            prefix="trj_" + label,
            suffix=(".bin" if self.parameters['binary_dump'] else ""),
            dir=tempdir,
            delete=(not self.parameters['keep_tmp_files']),
        )
        lammps_trj = lammps_trj_fd.name
        if self.parameters['no_data_file']:
            lammps_data = None
        else:
            lammps_data_fd = NamedTemporaryFile(
                prefix="data_" + label,
                dir=tempdir,
                delete=(not self.parameters['keep_tmp_files']),
                mode='w',
                encoding='ascii'
            )
            write_lammps_data(
                lammps_data_fd,
                self.atoms,
                specorder=self.parameters['specorder'],
                force_skew=self.parameters['always_triclinic'],
                reduce_cell=self.parameters['reduce_cell'],
                velocities=self.parameters['write_velocities'],
                prismobj=self.prism,
                units=self.parameters['units'],
                atom_style=self.parameters['atom_style'],
            )
            lammps_data = lammps_data_fd.name
            lammps_data_fd.flush()

        result_dict = {
            'tempdir': tempdir,
            'lammps_in': lammps_in,
            'lammps_log': lammps_log,
            'lammps_trj': lammps_trj,
            'lammps_data': lammps_data,
            'lammps_trj_fd': lammps_trj_fd,
            'lammps_data_fd': lammps_data_fd,
        }

        return result_dict

    def _run_write_input_and_get_output(self, **lammps_path_dict):
        tempdir = lammps_path_dict['tempdir']
        lammps_in = lammps_path_dict['lammps_in']
        lammps_log = lammps_path_dict['lammps_log']
        lammps_trj = lammps_path_dict['lammps_trj']
        lammps_data = lammps_path_dict['lammps_data']

        # see to it that LAMMPS is started
        if not self._lmp_alive():
            command = self.get_lammps_command()
            # Attempt to (re)start lammps
            self._lmp_handle = subprocess.Popen(
                shlex.split(command, posix=(os.name == "posix")),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # HM: Capture stderr for debugging
                encoding='ascii',
            )

        lmp_handle = self._lmp_handle

        # Create thread reading lammps stdout (for reference, if requested,
        # also create lammps_log, although it is never used)
        if self.parameters['keep_tmp_files']:
            lammps_log_fd = open(lammps_log, "w")
            fd_log = SpecialTee(lmp_handle.stdout, lammps_log_fd)
        else:
            fd_log = lmp_handle.stdout
        # thr_read_log = Thread(target=self.read_lammps_log, args=(fd_log,))
        # thr_read_log.start()

        # write LAMMPS input (for reference, also create the file lammps_in,
        # although it is never used)
        if self.parameters['keep_tmp_files']:
            lammps_in_fd = open(lammps_in, "w")
            fd = SpecialTee(lmp_handle.stdin, lammps_in_fd)
        else:
            fd = lmp_handle.stdin
        write_lammps_in(
            lammps_in=fd,
            parameters=self.parameters,
            atoms=self.atoms,
            prismobj=self.prism,
            lammps_trj=lammps_trj,
            lammps_data=lammps_data,
        )

        # Wait for log output to be read (i.e., for LAMMPS to finish)
        # and close the log file if there is one
        self.read_lammps_log(fd_log)
        # thr_read_log.join()

        if self.parameters['keep_tmp_files']:
            lammps_in_fd.close()
            lammps_log_fd.close()

        if not self.parameters['keep_alive']:
            self._lmp_end()

        exitcode = lmp_handle.poll()
        if exitcode and exitcode != 0:
            raise RuntimeError(
                "LAMMPS exited in {} with exit code: {}."
                "".format(tempdir, exitcode)
            )

        return lmp_handle

    def _run_check_success(self):

        is_success = len(self.thermo_content) != 0
        if not is_success:
            self._lmp_end()  # To rerun the process
            print("LAMMPS calculation failed. Retrying...")

        return is_success

    def _run_write_output(
            self, lmp_handle, set_atoms=False, **lammps_path_dict):
        lammps_trj = lammps_path_dict['lammps_trj']
        lammps_data_fd = lammps_path_dict['lammps_data_fd']
        lammps_trj_fd = lammps_path_dict['lammps_trj_fd']

        # A few sanity checks
        if len(self.thermo_content) == 0:
            raise RuntimeError("Failed to retrieve any thermo_style-output")

        if int(self.thermo_content[-1]["atoms"]) != len(self.atoms):
            # This obviously shouldn't happen, but if prism.fold_...() fails,
            # it could
            raise RuntimeError("Atoms have gone missing")

        trj_atoms = read_lammps_dump(
            infileobj=lammps_trj,
            order=self.parameters['atorder'],
            index=-1,
            prismobj=self.prism,
            specorder=self.parameters['specorder'],
        )

        if set_atoms:
            self.atoms = trj_atoms.copy()

        self.forces = trj_atoms.get_forces()
        # !TODO: trj_atoms is only the last snapshot of the system; Is it
        #        desirable to save also the inbetween steps?
        if self.parameters['trajectory_out'] is not None:

            # !TODO: is it advisable to create here temporary atoms-objects
            self.trajectory_out.write(trj_atoms)

        tc = self.thermo_content[-1]
        self.results["energy"] = convert(
            tc["pe"], "energy", self.parameters["units"], "ASE"
        )
        self.results["free_energy"] = self.results["energy"]
        self.results['forces'] = convert(self.forces.copy(),
                                         'force',
                                         self.parameters['units'],
                                         'ASE')
        stress = np.array(
            [-tc[i] for i in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]
        )

        # We need to apply the Lammps rotation stuff to the stress:
        xx, yy, zz, yz, xz, xy = stress
        stress_tensor = np.array([[xx, xy, xz],
                                  [xy, yy, yz],
                                  [xz, yz, zz]])
        stress_atoms = self.prism.tensor2_to_ase(stress_tensor)
        stress_atoms = stress_atoms[[0, 1, 2, 1, 0, 0],
                                    [0, 1, 2, 2, 2, 1]]
        stress = stress_atoms

        self.results["stress"] = convert(
            stress, "pressure", self.parameters["units"], "ASE"
        )

        lammps_trj_fd.close()
        if not self.parameters['no_data_file']:
            lammps_data_fd.close()
