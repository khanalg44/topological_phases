#!/usr/bin/env python

from pythtb import *
from scipy import *
from pylab import *
import ptbe

def main():
	#onsite energies
	Ecu = 0.0
	Eo = 0.3

	#hopping amplitude
	V = 1.0				# Cu-->O
	t = -0.25			# O -->O 

	lat = [[1,0],[0,1]] # 2d square latice 

	orb = [[0.0,0.0],[0.0,0.5],[0.0,0.5]]  # three orbitals per unit cell Cu in the (0,0) position

	my_model = tbmodel(2,2, lat, orb)

	my_model.set_sites([Ecu,Eo,Eo])


	my_model.add_hop(V, 0, 1, [0,0])  	# Cu --> O in the same unit cell
	my_model.add_hop(V, 0, 2, [0,0])	#-----------------
	my_model.add_hop(V, 0, 1, [-1,0])	# Cu --> O in the unit cell to the right hence R = [1,0]
	my_model.add_hop(V, 0, 2, [0,-1])	# Cu --> O in the unit cell to the bottom hence R = [0,1]
	
	my_model.add_hop(t, 1, 2, [0,0])	# O --> O in the same unit cell
	my_model.add_hop(t, 1, 2, [1,0])	# O --> O unit cell to the right 
	my_model.add_hop(t, 1, 2, [0,-1])	# O --> O unit cell to down under
	my_model.add_hop(t, 1, 2, [1,-1])	# O --> O unit cell to the right and down

	my_model.display()

	
	path = [[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
	k_lab=(r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$')

	kpts = k_path(path, 201)

	(k_vec, k_dist, k_node) = k_path(path, 201, my_model)

	eval=my_model.solve_all(k_vec)
	#eval=my_model.solve_all(kpts)

	#print eval, evec

	plot_file='CuO2_band.pdf'
	plot_title='Cu)2 plane'

	ptbe.plot_bsr(plot_file,plot_title,k_vec,k_dist,k_node,k_lab,eval)


	#plot(eval[0])
	#plot(eval[1])
	#plot(eval[2])
	#show()


if __name__ == '__main__':
	main()
