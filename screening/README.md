# CryoEtchSimulator
- can do adsorption and diffusion simulation for surfaces with multiple gases.
- uses ASE, LAMMPS, and PACKMOL.
- is designed for batch calculations.

1. AdsorptionSimulator
```python
class AdsorptionSimulator():
    # ...
    def run(self):
        '''
        generate the surface with a given amount of additive gases.
        (Ex. IF5 gas 1 ML on AFS surface)
        '''
        self._process_molecules('additive')

        '''
        generate the surface with a given amount of etchant gases.
        (Ex. HF 1 ea on AFS+IF5 surface)
        '''
        self._process_molecules('etchant')

        '''
        calculate the effective adsorption energy,
        which is the weighted average of adsorption energies.
        '''
        self._summarize_results()
    # ...
```

2. DiffusionSimulator
```python
    def run(self):
        '''
        Create a supercell of the slab to simulate diffusion.
        '''
        self._replicate_slab()
        '''
        Run NVT-MD for a given temperature and time.
        (Ex. 300K, 10 ps)
        '''
        self._run_md()
        '''
        Calculate the MSD, diffusion coefficient, and activation energy.
        '''
        self._summarize_results()
```

3. Other utilities
`CryoEtchSimulator/genCell/`: cell generator using PACKMOL.
