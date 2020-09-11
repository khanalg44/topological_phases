from pythtb import *
from scipy import *

def main():

	lat = [[1.0,0.0],[0.0,1.0]]
	r = 1.0

	angle = 2.0*pi/3.0
	orb = [[0.0,0.0],[0.0,r],[0.5,r*sin(angle)]]

	# hopping
	t0 = -1.0
	s = -0.4
	phi  =0.0

	#magnetic flux
	alpha = pi/4

	my_model = tbmodel(0,2,lat, orb)

	# now find eigen vectors for different alpha values
	n_phi = 12
	evec_all = zeros((n_phi,3,3),dtype=complex)

	for j in range(n_phi):

		phi = 2.0* pi*float(j)/float(n_phi)

		#compute hopping
		p = 2.0*pi/3.0
		t = [ t0+s*cos(phi), 
				t0 +s*cos(phi-p),
				t0 + s*cos(phi-2*p)]

		#include magnetic flux correction
		t[2] = t[2] * exp((1j)*alpha)
		

		]





	my_model.display()


if __name__ == '__main__':
	main()