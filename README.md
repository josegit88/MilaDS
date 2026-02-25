# MilaDS
Substructure identifier in galaxy clusters using the DS+ method, version of the code developed in Python

----

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Motivation

The study of substructures in galaxy clusters is essential for determining their dynamical state, assembly history, and overall evolution. In this repository, we present a Python implementation of our **DS+** method for the identification and characterization of substructures in galaxy clusters. This method is based on the projected positions and line-of-sight velocities of cluster galaxies, and represents an improvement and extension of the traditional approach introduced by [Dressler & Shectman (1988)](https://articles.adsabs.harvard.edu/pdf/1988AJ.....95..985D). For further details and technical specifications, please refer to our main paper: [DS+, link here](https://www.aanda.org/articles/aa/full_html/2023/01/aa45422-22/aa45422-22.html)


## Quick start

Input:
Basic information for galaxies in the cluster:, (X/kpc, Y/kpc, Vlos/km s^-1, redshift, others)

Output:
    
The program returns three arrays:
    - Individual information for each galaxy, organized by multiplicity within each DS+ group
    - Assignment information for DS+ groups, with no overlapping galaxies
    - A summary of the identified DS+ groups

Example of running program:

```
milaDS.DSp_groups(Xcoor=X_data, Ycoor=Y_data, Vlos=Vlos_data, Zclus=0.296, cluster_name="Cl1", nsims=100, Plim_P=10)
```

## Citations

Please acknowledge by citing the project and using the following DOI as reference:
J. Benavides, A. Biviano & M. Abadi (2023), [DS+ paper, link here](https://www.aanda.org/articles/aa/full_html/2023/01/aa45422-22/aa45422-22.html)

You may also use the the following BibTex:

```
@ARTICLE{2023A&A...669A.147B,
       author = {{Benavides}, Jos{\'e} A. and {Biviano}, Andrea and {Abadi}, Mario G.},
        title = "{DS+: A method for the identification of cluster substructures}",
      journal = {\aap},
     keywords = {galaxies: clusters: general, galaxies: groups: general, galaxies: kinematics and dynamics, Galaxy: abundances, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = jan,
       volume = {669},
          eid = {A147},
        pages = {A147},
          doi = {10.1051/0004-6361/202245422},
archivePrefix = {arXiv},
       eprint = {2212.00040},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...669A.147B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```

## Contact

For bugs or question please contact

> **José Benavides** [jose.astroph@gmail.com](jose.astroph@gmail.com)
