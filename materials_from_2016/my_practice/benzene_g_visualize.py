from pythtb import *
from scipy import *
from pylab import *

#atomic distance
r = 1.2

#onsite energy
ep = -0.4

#hopping amplitude
t = -0.25

#lat vector
lat = [[1.0,0.0],[0.0,1.0]]

#orbitals are at different angles and positions

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

(fig, ax) = my_model.visualize(0,1)

fig.savefig("visualize_benzene.pdf")

	