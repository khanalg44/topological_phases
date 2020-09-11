#!/usr/bin/env python
from __future__ import print_function # python3 style print

# Band structure of Haldane model

from pythtb import * # import TB model class
import matplotlib.pyplot as plt

# define setup of Haldane model
def set_model(Delta,t_0,tprime,rashba):

  lat=[[1.0,0.0],[0.0,1.0]]
  orb=[[0.0,0.0],[0.5,0.5]]
  my_model=tbmodel(2,2,lat,orb,nspin=2)
  my_model.set_sites([-Delta,Delta])

  # definitions of Pauli matrices
  sigma_x=np.array([0.,1.,0.,0])
  sigma_y=np.array([0.,0.,1.,0])
  sigma_z=np.array([0.,0.,0.,1])

  # spin-independent first neighbor
  my_model.add_hop(-t_0, 0, 0, [ 1, 0])
  my_model.add_hop(-t_0, 0, 0, [ 0, 1])
  my_model.add_hop( t_0, 1, 1, [ 1, 0])
  my_model.add_hop( t_0, 1, 1, [ 0, 1])

  # spin-dependent second neighbor
  my_model.add_hop(    tprime        , 1, 0, [ 1, 1])
  my_model.add_hop( 1j*tprime*sigma_z, 1, 0, [ 0, 1])
  my_model.add_hop( -1*tprime        , 1, 0, [ 0, 0])
  my_model.add_hop(-1j*tprime*sigma_z, 1, 0, [ 1, 0])

  # rashba couplings on first-neighbor bands, 90 degrees rotated
  my_model.add_hop( rashba*sigma_y, 0, 0, [ 1, 0],mode="add")
  my_model.add_hop(-rashba*sigma_x, 0, 0, [ 0, 1],mode="add")
  my_model.add_hop( rashba*sigma_y, 1, 1, [ 1, 0],mode="add")
  my_model.add_hop(-rashba*sigma_x, 1, 1, [ 0, 1],mode="add")

  return my_model

# set model parameters and construct bulk model
t_0=1.0
tprime=0.4
param_sets=[[3.0,0.0], [3.0,0.2], [5.0,0.2]]  #  [Delta, rashba]

# set up figures
fig,ax=plt.subplots(1,3,figsize=(11,4))
labs=['(a)','(b)','(c)']

# run over three choices of tprime and comput hybrid Wannier center flow
for j,params in enumerate(param_sets):

  Delta=params[0]
  rashba=params[1]
  my_model=set_model(Delta,t_0,tprime,rashba)
  nk=51
  my_array=wf_array(my_model,[nk,nk])
  my_array.solve_on_grid([0.,0.])
  rbar = my_array.berry_phase([0,1],1,berry_evals=True,contin=True)/(2.*np.pi)
  
  k0=np.linspace(0.,1.,nk)
  ax[j].set_xlim(0.,1.)
  ax[j].set_ylim(-1.3,1.3)
  ax[j].set_xlabel(r"$\kappa_1/2\pi$")
  ax[j].set_ylabel(r"HWF centers")
  for shift in (-2.,-1.,0.,1.):
    ax[j].plot(k0,rbar[:,0]+shift,color='k')
    ax[j].plot(k0,rbar[:,1]+shift,color='k')
  ax[j].text(-.30,1.15,labs[j],size=20.)
  
# save figure as a PDF

fig.tight_layout()
plt.subplots_adjust(left=0.15,wspace=0.6)
fig.savefig("checkerboard-wflow.pdf")
