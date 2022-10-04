# MilaDS
Substructure identificator using the DS+ method

Input:
Basic information of galaxies into the cluster, (X,Y,Vlos, redshift, others)

Output:
    
Return three arrays:
    - individual information of each galaxy per multiplicity in each DS+ group
    - information of assignment of DS+ groups, with no-overlaping galaxies
    - summary of DS+ identified groups

Example of running program:

Example running:

milaDS.DSp_groups(Xcoor=data_sample[:,1], Ycoor=data_sample[:,2], Vlos=data_sample[:,3], Zclus=0.296, cluster_name="Cl1", nsims=100, Plim_P=10)

