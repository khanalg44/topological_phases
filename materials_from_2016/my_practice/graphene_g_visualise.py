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

#visualise

(fig,ax) = my_model.visualize(0,1)
fig.savefig("visualize_bulk.pdf")

cut_one = my_model.cut_piece(8,0,glue_edgs=False) # glue_edges=False is default value

(fig,ax) = cut_one.visualize(0,1)
fig.savefig('visualize_ribbon.pdf')


cut_two = cut_one.cut_piece(8,1,glue_edgs=False)

(fig,ax) = cut_two.visualize(0,1)
fig.savefig('visualize_finite.pdf')

print 'Done.'