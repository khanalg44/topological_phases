#!/usr/bin/env python

# two dimensional hopping nearest neighbor

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import * # import TB model class
import pylab as pl

# specify model
lat=[[1,0,0],[0,1,0],[0,0,1]] #specifies the lattice vectors
orb=[[0,0,0]] #displacement from center for each lattice site
my_model=tb_model(3,3,lat,orb) #constructs the tight binding model 
my_model.set_hop(1, 0, 0, [1, 0, 0]) #set hopping in the x-direction
my_model.set_hop(1, 0, 0, [0, 1, 0]) #set hopping in the y-direction
my_model.set_hop(1, 0, 0, [0, 0, 1]) #set hopping in the z-drection
print "Model defined for 3D lattice."

# #solve the model in the x-direction
# pathX = [[0,0,0],[1,0,0]] #testing the diagonal
# kptsX = k_path(pathX, 100) #creates a bunch of points for the path
# evalX = my_model.solve_all(kptsX) #solves the model
# print "Model solved in x-direction."

# #solve the model in the y-direction
# pathY = [[0,0,0],[0,1,0]] #testing the diagonal
# kptsY = k_path(pathY, 100) #creates a bunch of points for the path
# evalY=my_model.solve_all(kptsY) #solves the model
# print "Model solved in y-direction."

# #solve the model in the z-direction
pathZ = [[0,0,0],[0,0,1]] #testing the diagonal
kptsZ = k_path(pathZ, 100) #creates a bunch of points for the path
evalZ = my_model.solve_all(kptsZ) #solves the model
print "Model solved in z-direction."

#solve the model in the diagonal direction
#pathDiag = [[0,0,0],[1,1,1]] #testing the diagonal
#kptsDiag=k_path(pathDiag, 100) #creates a bunch of points for the path
#evalDiag=my_model.solve_all(kptsDiag) #solves the model
#print "Model solved in the diagonal direction."

# plot band structure
fig=pl.figure()
# pl.plot(evalX[0])
# pl.plot(evalY[0])
# pl.plot(evalZ[0])
pl.plot(evalZ[0])
pl.title("3D chain band structure")
pl.xlabel("Path in k-space")
pl.ylabel("Band energy")
pl.savefig("simple_band3d.pdf")
