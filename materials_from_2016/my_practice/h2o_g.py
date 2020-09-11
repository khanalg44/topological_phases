from pythtb import *
from scipy import *
from pylab import *


b = 1.0
angle = 54*pi/180 # half angle
E_os = -1.5 
E_op = -1.2
E_h = -1.0

#hopping [O(s)-H(s), O(p)-H(s)]
ts = -0.3
tp = -0.4

lat=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]

#orb = [[0,0],[0,0],[0,0],[0,0], 
#	[b*sin(angle), b*cos(angle)],[-b*sin(angle), b*cos(angle)]]

orb=[ [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],
      [b*np.cos(angle), b*np.sin(angle),0.],
      [b*np.cos(angle),-b*np.sin(angle),0.] ]


my_model = tbmodel(0,3,lat, orb)

my_model.set_onsite([E_os, E_op, E_op, E_op, E_h, E_h])

my_model.set_hop(ts, 0, 4 )
my_model.set_hop(ts, 0, 5 )
my_model.set_hop(tp*cos(angle), 1, 4 )
my_model.set_hop(tp*cos(angle), 1, 5 )
my_model.set_hop(tp*cos(angle), 2, 4 )
my_model.set_hop(-tp*cos(angle), 2, 5 )

my_model.display()

#path=[[0.0,0.0,0.0],[0.0,0.0,0.5],[0.5,0.5,0.5,0.5],[0.0,0.0,0.0]]
path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
kpts = k_path(path,201)


eval = my_model.solve_all(kpts)

plot(eval)