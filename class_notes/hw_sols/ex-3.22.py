#!/usr/bin/env python
from __future__ import print_function # python3 style print

# Chain with three sites per cell

from pythtb import *
import matplotlib.pyplot as plt

# define function to construct model
def set_model(t,delta,lmbd):
    lat=[[1.0]]
    orb=[[0.0],[1.0/3.0],[2.0/3.0]]
    model=tb_model(1,1,lat,orb)
    model.set_hop(t, 0, 1, [0])
    model.set_hop(t, 1, 2, [0])
    model.set_hop(t, 2, 0, [1])
    onsite_0=delta*(-1.0)*np.cos(2.0*np.pi*(lmbd-0.0/3.0))
    onsite_1=delta*(-1.0)*np.cos(2.0*np.pi*(lmbd-1.0/3.0))
    onsite_2=delta*(-1.0)*np.cos(2.0*np.pi*(lmbd-2.0/3.0))
    model.set_onsite([onsite_0,onsite_1,onsite_2])
    return(model)

# construct the model
t=-1.3
delta=2.0
lmbd=0.3
my_model=set_model(t,delta,lmbd)

# construct finite model by cutting 10 cells from infinite chain
finite_model=my_model.cut_piece(10,0)

# define function to return first 10 Wannier centers
def print_centers(finite_model):
  (feval,fevec)=finite_model.solve_all(eig_vectors=True)
  xbar0=finite_model.position_hwf(fevec[0:10,],0)
  xbar1=finite_model.position_hwf(fevec[10:20,],0)
  xbarb=finite_model.position_hwf(fevec[0:20,],0)
  print ("\nFinite-chain Wannier centers associated with band 0:")
  print((10*"%7.4f")% tuple(xbar0))
  print ("\nFinite-chain Wannier centers associated with band 1:")
  print((10*"%7.4f")% tuple(xbar1))
  print ("\nFirst 10 finite-chain Wannier centers for both bands:")
  print((10*"%7.4f")% tuple(xbarb[0:10]))

print("\nUnmodified at chain end")
print_centers(finite_model)

print("\nModified site energies at chain end")
finite_model.set_onsite(-1.3,0,mode="add")
finite_model.set_onsite(+0.5,1,mode="add")
print_centers(finite_model)

print("\nAlso modified hopping at chain end")
finite_model.set_hop(1.8,0,1,[0],mode="add")
print_centers(finite_model)
