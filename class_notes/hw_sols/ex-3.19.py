#!/usr/bin/env python
from __future__ import print_function # python3 style print

# Homework 3.19
# s-px chain model

from pythtb import *
import matplotlib.pyplot as plt

def main():

    # set model parameters
    deltaE = 8.0  # difference between s and p orbital energies
    t      = 0.8  # strength of bonds; adjust this and see what happens

    # set up model
    lat=[[1.0]]
    orb=[[0.0],[0.0]]
    my_model=tbmodel(1,1,lat,orb)
    my_model.set_onsite([-deltaE/2,deltaE/2])
    (V_ss, V_pp, V_sp)=(-1.40*t, 3.24*t, 1.84*t)
    my_model.set_hop( V_ss, 0, 0, [1])
    my_model.set_hop( V_sp, 0, 1, [1])
    my_model.set_hop(-V_sp, 1, 0, [1])
    my_model.set_hop( V_pp, 1, 1, [1])

    # print tight-binding model
    my_model.display()

    # construct the k-path
    (k_vec,k_dist,k_node)=my_model.k_path('full',121)
    k_lab=(r'0',r'$\pi$',r'$2\pi$')

    # solve for eigenenergies of hamiltonian
    evals=my_model.solve_all(k_vec)
    
    # plotting of band structure
    print('Plotting bandstructure...')

    fig, ax = plt.subplots(figsize=(4.,3.))
    ax.set_xlim([0,k_node[-1]])
    ax.set_xticks(k_node)
    ax.set_xticklabels(k_lab)
    ax.axvline(x=k_node[1],linewidth=0.5, color='k')
    ax.set_xlabel("k")
    ax.set_ylabel("Band energy")

    # plot first and second bands
    ax.plot(k_dist,evals[0],color='k')
    ax.plot(k_dist,evals[1],color='k')
    
    # save figure as a PDF
    fig.tight_layout()
    #fig.savefig("sp_1d.pdf")
    plt.show()

    print('Compute and print Berry phase.\n')

    evec_array=wf_array(my_model,[121])
    evec_array.solve_on_grid([0.])
    print("Berry phase is %7.3f"% evec_array.berry_phase([0]))

    print('Done.\n')

main()
