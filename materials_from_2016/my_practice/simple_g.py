#!/usr/bin/env python

from pythtb import *
from scipy import *
#from matplotlib import pyplot as pl
import pylab as pl

#specify model
lat = [[1.0]]
orb = [[0.0]]
my_model = tb_model(1,1,lat,orb)
my_model.set_hop(-1., 0, 0, [1])

#solve model
path = [[-0.5],[0.5]]
kpts = k_path(path,100)
evals=my_model.solve_all(kpts)
#evals=my_model.visualize(kpts)

#plot band structure
fig=pl.figure()
pl.plot(evals[0])
pl.title("1D chain band structure")
pl.xlabel("Path in k-space")
pl.ylabel("Band Energy")
pl.savefig("simple_band.pdf")

#pl.plot(x, f(x))

pl.show()
savetxt('aa.dat',evals[0])

#csa@edc.com.np
#csa@edc.com.np
