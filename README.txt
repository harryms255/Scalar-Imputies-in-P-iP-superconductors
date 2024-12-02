This is a set of functions to calculate the electronic properties of an interface of scalar impurities embedded into a 2D p+ip superconductor. This includes symmetry analysis, numeric calculations of tight binding models, analytic Green's function calculations, calculation of in-gap bands and local density of states. Topological properties can be calculated from the bulk Pfaffian, the Pfaffian of the topological Hamiltonian and through spectral localiser theory to determine topological properties of the mixed dimension phase.

Descriptions of the parameters used are present in the file p_ip_functions_file.py. 

The following non-standard modules will need to be installed

Qsymm (for symmetry analysis)
    Can be installed using 'conda install conda-forge::qsymm' or 'pip install qsymm'

Pfapack (for efficient calculation of pfaffians)
    Can be installed using 'conda install pfapack' or 'pip install pfapack'
