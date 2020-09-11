#!/use/bin/env python                                                        

######################################
#
# Tight binding calculation for 2d FeSe 
#
#######################################

from pythtb import *
import ptbe
from pylab import *

def main():
    lat=[[1,0],[0,1]]

    orb=[[0,0.5],[0.5,0],[1,0.5],[0.5,1]]	# Se, Fe, Se, Fe

    my_model=tbmodel(2,2,lat,orb)

    #set onsite
    E_fe=0.8
    E_se=0.4
    my_model.set_onsite([E_se, E_fe, E_se, E_fe])

    #set hopping
    t=0.4
    my_model.set_hop(t, 0,1,[0,0])
    my_model.set_hop(t, 0,3,[0,0])
    my_model.set_hop(t, 0,1,[-0.5,0.5])
    my_model.set_hop(t, 0,3,[-0.5,-0.5])

    my_model.set_hop(t, 1,2,[0,0])
    my_model.set_hop(t, 1,2,[-0.5,-0.5])
    #my_model.set_hop(t, 1,0,[0.5,-0.5])

    my_model.set_hop(t, 2,3,[0,0])
    my_model.set_hop(t, 2,3,[0.5,-0.5])
    #my_model.set_hop(t, 2,1,[0.5,0.5])
    #my_model.set_hop(t, 3,0,[0.5,0.5])
    #my_model.set_hop(t, 3,2,[-0.5,0.5])

    my_model.display()

    
    path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
    #path=[[0.0,0.5],[0.5,0.5],[0.0,0.0]]

    (k_vec, k_dist, k_node)=k_path(path, 100, my_model)
    k_lab=(r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$')

    evals =my_model.solve_all(k_vec)
    #print evalsX
    #plot(kptsX, evalsX[3])
    #plot(kptsX, evalsX[0])
    #show()
    #plot_file='FeSe_plane_X_M_G.pdf'
    plot_file='FeSe_plane.pdf'
    plot_title='FeSe plane Band structure'

    ptbe.plot_bsr(plot_file, plot_title, k_vec, k_dist, k_node, k_lab,evals)

    print 'Done.\n'


if __name__ == '__main__':
	main()
