#!/usr/bin/env python

# ----------------------------------------------------------
# tight-binding model for p_z states of benzene molecule
# ----------------------------------------------------------

#import the pythtb module
from pythtb import *

# distance of atoms from center
r=1.2

# site energy
ep=-0.4

# hopping
t=-0.25

# define frame for defining vectors: 2D Cartesian
lat=[[1.0,0.0],[0.0,1.0]]

# define coordinates of orbitals:
orb=np.zeros((6,2),dtype=float)
for i in range(6):
  angle=i*np.pi/3.0
  orb[i,:]= [r*np.cos(angle), r*np.sin(angle)]

my_model=tbmodel(0,2,lat,orb)

my_model.set_onsite([ep,ep,ep,ep,ep,ep])
my_model.set_hop(t,0,1)
my_model.set_hop(t,1,2)
my_model.set_hop(t,2,3)
my_model.set_hop(t,3,4)
my_model.set_hop(t,4,5)
my_model.set_hop(t,5,0)

my_model.display()

(eval,evec)=my_model.solve_all(eig_vectors=True)

# my_print(eval,evec)

n=len(eval)
evecr=evec.real

print "  n   eigval   eigvec"
for i in range(n):
  print " %2i  %7.3f" % (i,eval[i]),
  print "  ("+", ".join("%6.2f" % x for x in evecr[i,:])+" )"
