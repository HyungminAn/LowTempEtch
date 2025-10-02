This code calculates the adsorption energy of the given etchant molecule.

Prerequisite:
    1) relaxed molecule structure
    2) relaxed slab structure

Pipeline consists of:
    1) Make a relaxed structure with additive:
        1-1) insert molecule at random position on slab
        1-2) relaxation of the structure
        1-3) select the slab structure with the lowest energy

    2) Relax the structure with a etchant molecule:
        2-1) insert molecule at random position on slab
        2-2) relaxation of the structure
        2-3) Check for the dissociation of the etchant molecule
             The number of dissociation --> *chemisorption_ratio*
        2-4) Check for surface reconstruction
        2-5) For reconstructed structures, Remove the inserted etchant molecule
        2-6) Re-relax for the new slab structure
        2-7) Re-check for the slab reconstruction

    3) Gather the results and analyze the data:
        3-1) Get the chemisorption ratio
        3-2) Get the physisorption ratio
             = total - chemisorption ratio - reconstruction ratio
        3-3) Get the statistics value for adsorption energy
            : count, average, max, min, stddev, ...
