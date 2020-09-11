from scipy import *
from pylab import *

def E_k00(Es,t,a,k):
	return Es + 8*t*cos(k*a/2)

def E_kkk(Es,t,a,k):
	return Es + 6*t*cos(k*a/2) + 2*t*cos(3*k*a/2)


if __name__ == '__main__':
	Es = 4.5 # in ev
	t = -1.4 # in ev
	a  =3.5
	k1 = linspace(0,2*pi/a, 100)
	k2 = linspace(-pi/a,0, 100)

	plot(k1, E_k00(Es,t,a,k1),k2, E_kkk(Es,t,a,k2))
	xlabel('k')
	ylabel('E')
	title('Dispersion relation for Li, E(k,0,0) for (0,2 pi) and E(k,k,k) for (-pi/2a,0)')
	show()
