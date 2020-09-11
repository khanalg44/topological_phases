#!/usr/bin/env python

# ----------------------------------------------------------
# tight-binding model for H2O molecule
# ----------------------------------------------------------

#import the pythtb module
from pythtb import *
from pylab import *

# geometry: bond length and half bond-angle
b=1.0; angle=54.0*np.pi/180

# site energies [O(s), O(p), H(s)]
eos=-1.5; eop=-1.2; eh=-1.0

# hoppings [O(s)-H(s), O(p)-H(s)]
ts=-0.4; tp=-0.3

# define frame for defining vectors: 3D Cartesian
lat=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]

# define coordinates of orbitals: O(s,px,py,pz) ; H(s) ; H(s)
orb=[ [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],
      [b*np.cos(angle), b*np.sin(angle),0.],
      [b*np.cos(angle),-b*np.sin(angle),0.] ]

my_model=tbmodel(0,3,lat,orb)

my_model.set_onsite([eos,eop,eop,eop,eh,eh])
my_model.set_hop(ts,0,4)
my_model.set_hop(ts,0,5)
my_model.set_hop(tp*np.cos(angle),1,4)
my_model.set_hop(tp*np.cos(angle),1,5)
my_model.set_hop(tp*np.sin(angle),2,4)
my_model.set_hop(-tp*np.sin(angle),2,5)

my_model.display()

(eval,evec)=my_model.solve_all(eig_vectors=True)

# my_print(eval,evec)

n=len(eval)
evecr=evec.real

# signs of evec's are regularized by the following rule
# (this is arbitrary and not very important)
for i in range(n):
  if sum(evecr[i,1:4]) < 0:
    evecr[i,:]=-evecr[i,:]

print "  n   eigval   eigvec"
for i in range(n):
  print " %2i  %7.3f" % (i,eval[i]),
  print "  ("+", ".join("%6.2f" % x for x in evecr[i,:])+" )"
