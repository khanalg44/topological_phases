#!/usr/bin/env python

# ----------------------------------------------------------
# tight-binding model for trimer with possible magnetic flux
# ----------------------------------------------------------

#import the pythtb module
from pythtb import *

#import pythtb extensions
import ptbe

def main():
    # define frame for defining vectors: 2D Cartesian
    lat=[[1.0,0.0],[0.0,1.0]]
    
    # distance of atoms from center
    r=1.0
    
    # math constants
    tpio3=2.0*np.pi/3.0
    tpio4=2.0*np.pi/4.0

    # define coordinates of orbitals:
    orb=np.zeros((3,2),dtype=float)
    for i in range(3):
      angle=tpio4+(i+1)*tpio3
      orb[i,:]= [r*np.cos(angle), r*np.sin(angle)]
    
    # hopping parameters
    t0 =-1.0
    s  =-0.4
    phi=0.

    # numpy print precision
    np.set_printoptions(precision=4)

    # set up model (leave site energies set to zero)
    my_model=tbmodel(0,2,lat,orb)

    # loop over alpha values
    n_alpha=361
    exp_berry=np.zeros(n_alpha,dtype=complex)
    for j_alpha in range(n_alpha):

        # magnetic flux (alpha=2pi is one flux quantum)
        alpha=2.0*np.pi*float(j_alpha)/float(n_alpha-1)

        # set up array of wavefunctions as phi ranges though 0 to 2pi
        n_phi=24
        evec_array=np.zeros((n_phi,3),dtype=complex)
    
        # run over values of phi and fill the array
        for j_phi in range(n_phi):
    
            phi=2.0*np.pi*float(j_phi)/float(n_phi)
            # compute hoppings
            t=[ t0+s*np.cos(phi) ,
                t0+s*np.cos(phi-tpio3),
                t0+s*np.cos(phi-2.0*tpio3) ]
    
            # magnetic flux correction
            t[2]=t[2]*np.exp((1.j)*alpha)
    
            # set up model (leave site energies at zero)
            my_model=tbmodel(0,2,lat,orb)
            my_model.set_hop(t[0],0,1)
            my_model.set_hop(t[1],1,2)
            my_model.set_hop(t[2],2,0)
    
            # compute eigenvalues and eigenvectors
            (eval,evec)=my_model.solve_all(eig_vectors=True)
            evec_array[j_phi,:]=evec[0,:]
            
        # compute Berry phase
        b_phase=0.
        for j in range(n_phi):
            k=(j+1)%n_phi
            z=np.dot(evec_array[j,:].conjugate(),evec_array[k,:])
            b_phase=b_phase-np.angle(z)
    
        # record result as complex exponential
        exp_berry[j_alpha]=np.exp(-1.j*b_phase)

    # convert from exponential to ordinary form
    berry=np.zeros(n_alpha)
    berry[0]=np.angle(exp_berry[0])
    for j in range(1,n_alpha):
        x=np.angle(exp_berry[j]/exp_berry[j-1])
        berry[j]=berry[j-1]+x

    # now plot the results
    fig=pl.figure()
    pl.plot(berry)
    pl.xlim(0.,360.)
    pl.title("Berry phases for pseudorotation of trimer")
    pl.xlabel("Magnetic flux through the triangle")
    pl.ylabel("Berry phase")
    pl.savefig("trimer-loop.pdf")
main()
