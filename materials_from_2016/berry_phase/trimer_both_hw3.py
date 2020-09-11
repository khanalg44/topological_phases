#!/usr/bin/env python
# ----------------------------------------------------------
# tight-binding model for trimer with possible magnetic flux
# ----------------------------------------------------------

from pythtb import * #import the pythtb module
import ptbe #import pythtb extensions

def main():
    # define frame for defining vectors: 2D Cartesian
    lat=[[1.0,0.0],[0.0,1.0]]
    r=1.0 # distance of atoms from center
    tpio3=2.0*np.pi/3.0; tpio4=2.0*np.pi/4.0

    # define coordinates of orbitals:
    orb=np.zeros((3,2),dtype=float)
    for i in range(3):
      angle=tpio4+(i+1)*tpio3
      orb[i,:]= [r*np.cos(angle), r*np.sin(angle)]
    
    # hopping parameters
    t0 =-1.0; s  =-0.4;   phi=0.

    # magnetic flux (alpha=2pi is one flux quantum)
    alpha=np.pi/4.
    
    # numpy print precision
    np.set_printoptions(precision=4)

    # set up model (leave site energies set to zero)
    #dim_k=0 as we are only working on one unit cell.
    my_model=tbmodel(0,2,lat,orb)

    # set up array of wavefunctions as phi ranges though 0 to 2pi
    # do it two ways:
    #   (a) using class wf_array
    #   (b) by hand
    n_phi=12
    
    evec_all=np.zeros((n_phi,3,3),dtype=complex)

    # run over values of phi and fill the array
    for j in range(n_phi):

        phi=2.0*np.pi*float(j)/float(n_phi)
        # compute hoppings
        t=[ t0+s*np.cos(phi) ,
            t0+s*np.cos(phi-tpio3),
            t0+s*np.cos(phi-2.0*tpio3) ]

        # magnetic flux correction
        t[2]=t[2]*np.exp((1.j)*alpha)

        # set hoppings
        my_model.set_hop(t[0],0,1,mode="reset")
        my_model.set_hop(t[1],1,2,mode="reset")
        my_model.set_hop(t[2],2,0,mode="reset")

        # print out model on first pass only
        if j == 0: my_model.display()
        
        # compute eigenvalues and eigenvectors
        (eval,evec)=my_model.solve_all(eig_vectors=True)
        evec_all[j,:,:]=evec
        
        # print results from ground state
        print "phi =%6.3f"%phi,"eval =%8.4f"%eval[0],"evec =",evec[0,:]

    # ---------------------------------------------------------------
    # compute Berry phase by hand for lowest band
    # ---------------------------------------------------------------

    b_phase=0.
    for j in range(n_phi):
        k=(j+1)%n_phi
        z=np.dot(evec_all[j,0,:].conjugate(),evec_all[k,0,:])
        b_phase=b_phase-np.angle(z)

    # make branch choice such that b_phase lies in (0,2pi)
    t=b_phase+np.pi
    t=t%(2.*np.pi)   # this is python 'modulo' syntax
    b_phase=t-np.pi

    #print final result
    print "Berry phase calc by hand is   ","%8.4f"%b_phase

    # ---------------------------------------------------------------
    # the following is an alternate solution using wf_array functions
    # ---------------------------------------------------------------

    # we add one extra point; the first and last will be identical
    n=n_phi+1

    # create empty array belonging to 'wf_array' class
    evec_wf=wf_array(my_model,[n])

    # fill values
    for j in range(n-1):
        evec_wf[j]=evec_all[j]
    # and the last point is the same as the first
    evec_wf[-1]=evec_wf[0]

    # compute and print the open-path berry phase for the lowest band
    b_phase=evec_wf.berry_phase([0])
    print "Berry phase using wf_array is ","%8.4f"%b_phase

main()
