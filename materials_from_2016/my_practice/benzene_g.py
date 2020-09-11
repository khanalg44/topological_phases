from pythtb import *
from scipy import *
from pylab import *


# benzene has hexagonal strcture
# the hopping is from 0-> 1, 1->2 2->3 ... 5-> 0 
# dim_k = 0 hence R = 0 Meaning can't hop to other unit cells (I guess!)
# hence we won't plot a dispersion relation
# 


r = 1.2 #atomic distance
ep = -0.4 #onsite energy
t = -0.25 #hopping amplitude

lat = [[1.0,0.0],[0.0,1.0]]
orb = zeros((6,2),dtype = float)

for i in range(6):
    angle = i * pi/3   # origin at the center of the Benzene
    orb[i,:] = [r* cos(angle), r*sin(angle)] 
    # A[i,:] : will take all the value of that row A[0,:] will be entire first row
    
my_model = tbmodel(0,2,lat, orb)

my_model.set_onsite([ep,ep,ep,ep,ep,ep])

my_model.set_hop(t,0,1)
my_model.set_hop(t,1,2)
my_model.set_hop(t,2,3)
my_model.set_hop(t,3,4)
my_model.set_hop(t,4,5)
my_model.set_hop(t,5,0)


my_model.display()

(eval,evec) = my_model.solve_all(eig_vectors=True)

print eval, evec

n = len(eval)

ev = evec.real

for i in range(n):
	print "%2i, %5.4f" % (i, eval[i]),
	#print " ( " + ",".join("%6.2f" % x for x in ev[i,:]+")"