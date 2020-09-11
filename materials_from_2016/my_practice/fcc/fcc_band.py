#!/usr/bin/env python
from pythtb import *
from pylab import *

#2D FCC Lattice
#Atom A in the corner and atom B in the face center

lat=[[1,0],[0,1]]
orb=[[0,0],[0.5,0.0],[0.75,0.5],[0.25,0.5]]

t1=1.0 # hopping A-B
t2=1.0 # hoppinf B-B
my_model= tb_model(2,2,lat,orb)

E_o=0.0
E_cu=0.5

#my_model.set_onsite([E_cu,E_o,E_o])
my_model.set_hop(t1,0,1,[0,0])
my_model.set_hop(t1,0,2,[0,0])
my_model.set_hop(t1,0,3,[-1,0])
my_model.set_hop(t1,0,3,[-1,-1])
my_model.set_hop(t1,0,2,[0,-1])

my_model.set_hop(t1,1,2,[0,0])
my_model.set_hop(t1,1,3,[0,0])
my_model.set_hop(t1,1,3,[0,-1])
my_model.set_hop(t1,1,2,[0,-1])

# solve in x direction

figure(1)
pathX=[[0,0],[1,0]]  	# path gamma - X points
kptsX=k_path(pathX, 50) 	# kpts in above defined path
evals=my_model.solve_all(kptsX)
#print evals
#print len(evals[0])
plot(kptsX,evals[3],label='orb_3')
plot(kptsX,evals[2],label='orb_2')
plot(kptsX,evals[1],label='orb_1')
plot(kptsX,evals[0],label='orb_0')
legend(loc='best')

print evals

print size(evals)

show()

# solve in y direction

#pathX=[[0,0,0],[1,1,1]]  	# path in k space in x direction
#kptsX=k_path(pathX, 201) 	# kpts in above defined path
#evals=my_model.solve_all(kptsX)
#print evals
#print len(evals[0])
#plot(kptsX,evals[2])
#show()


