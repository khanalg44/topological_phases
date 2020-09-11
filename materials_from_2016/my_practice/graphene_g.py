#!/usr/bin/env python


from pythtb import *
from scipy import *
from pylab import *
from ptbe import *

#set parameters

lat = [[1.0,0.0],[0.5, sqrt(3)/2]]
orb = [[1./3.,1./3.],[2./3. ,2./3.]]
dim_r = 2
dim_k = 2

my_model = tb_model(dim_k, dim_r, lat, orb)

#set hoppings
delta = 0.0
t = -1.0

my_model.set_onsite([-delta, delta])

my_model.set_hop(t, 0,1,[0,0])  #on same unit cell
my_model.set_hop(t, 1,0,[0,1])	# on a unit cell of neighbour of distance R
my_model.set_hop(t, 1,0,[1,0])	#on another unit cell of dist T

my_model.display()

path = [[1.0,0.0],[0.0,1.0]]
kpts= k_path(path,200)

print kpts

#calculations
(eval,evec) = my_model.solve_all(kpts)
print_eig_real(eval,evec)
#print evals[0]

#plot(evals[0])
#plot(evals[1])
#show()
