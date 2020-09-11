from pythtb import *
from scipy import *
from pylab import *


lat = [[1.0]]

orb = [[0.0],[1.0/3.0],[2.0/3.0]]  # I guess for a 1-d chain with three orbital per unit cell

my_model = tb_model(1,1,lat,orb)

delta = 2.0
t  = -1.

#hopping
# -|--0---1---2-|--0---1---2-|--0---0
# hopping from 2 -> 0 will be in a different unit cell hence R = 1

my_model.set_hop(t,0,1,[0])
my_model.set_hop(t,1,2,[0])
my_model.set_hop(t,2,0,[1]) 


my_model.display()

fig_onsite=figure()
ax_onsite=fig_onsite.add_subplot(111)

fig_band = figure()
ax_band = fig_band.add_subplot(111)

#no of steps from initial to final steps in which lambda changes
path_steps = 20 

all_lambda = linspace(0.0,1.0,path_steps, endpoint=True)

num_kpt = 30   # why number of k-points =30?


wf_kpt_lambda=wf_array(my_model, [num_kpt, path_steps])

for i_lambda in range(path_steps):
	lmbd = all_lambda[i_lambda]
	onsite_0 = delta*(-1.0)*cos(2.0*pi*(lmbd-0.0/3.0))
	onsite_1 = delta*(-1.0)*cos(2.0*pi*(lmbd-1.0/3.0))
	onsite_2 = delta*(-1.0)*cos(2.0*pi*(lmbd-2.0/3.0))

	my_model.set_onsite([onsite_0, onsite_1, onsite_2],mode='reset')

	k_vals = k_path([[-0.5],[0.5]],num_kpt,endpoint=True)
	(eval,evec) = my_model.solve_all(k_vals, eig_vectors=True)

	for i_kpts in range(num_kpt):
		wf_kpt_lambda[i_kpts,i_lambda]=evec[:,i_kpts,:]

	ax_onsite.scatter([lmbd],[onsite_0],c="r")
	ax_onsite.scatter([lmbd],[onsite_1],c="g")
	ax_onsite.scatter([lmbd],[onsite_2],c="b")

	col = 1.0-0.5*lmbd

	for band in range(eval.shape[0]):
		ax_band.plot(ravel(k_vals),eval[band,:],'-',color=[col,col,col])

fig_onsite.savefig("lambda_onsite.pdf")