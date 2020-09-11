#!/usr/bin/env python

# Homework 2.3
# two dimensional tight-binding CuO2 plane model

#import the pythtb module
from pythtb import *

#import pythtb extensions
import ptbe

def main():
    # define lattice vectors
    lat=[[1.0,0.0],[0.0,1.0]]
    # define coordinates of orbitals
    orb=[[0.0,0.0],[0.5,0.0],[0.0,0.5]]

    # make two dimensional tight-binding checkerboard model
    my_model=tbmodel(2,2,lat,orb)

    # set model parameters
    E_0=0.3
    V=1.0
    t=-0.25

    # set on-site energies
    my_model.set_sites([0.0,E_0,E_0])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    my_model.add_hop(V, 0, 1, [ 0, 0])
    my_model.add_hop(V, 0, 2, [ 0, 0])
    my_model.add_hop(V, 0, 1, [-1, 0])
    my_model.add_hop(V, 0, 2, [ 0,-1])
    my_model.add_hop(t, 1, 2, [ 0, 0])
    my_model.add_hop(t, 1, 2, [ 1, 0])
    my_model.add_hop(t, 1, 2, [ 0,-1])
    my_model.add_hop(t, 1, 2, [ 1,-1])

    # print tight-binding model
    my_model.display()

    # generate list of k-points following some high-symmetry line in
    # the k-space. Variable kpts here is just an array of k-points
    path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
    k_lab=(r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$')

    (k_vec,k_dist,k_node)=k_path(path,201,my_model)

    print '---------------------------------------'
    print 'starting calculation'
    print '---------------------------------------'
    print 'Calculating bands...'

    # solve for eigenenergies of hamiltonian on
    # the set of k-points from above
    evals=my_model.solve_all(k_vec)
    
    # plotting of band structure
    print 'Plotting bandstructure...'

    plot_file="cuo2-plane.pdf"
    plot_title="CuO2 plane band structure"

    ptbe.plot_bsr(plot_file,plot_title,k_vec,k_dist,k_node,k_lab,evals)

    print 'Done.\n'

if __name__ == '__main__':
    main()
