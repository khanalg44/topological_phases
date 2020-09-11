from pythtb import *

# lattice vectors and orbital positions
lat=[[1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]]
orb=[[1./3., 1./3.], [2./3., 2./3.]]
graphene=tb_model(2,2, lat, orb)

# define hopping between orbitals
graphene.set_hop(-1.0, 0, 1, [ 0, 0])
graphene.set_hop(-1.0, 1, 0, [ 1, 0])
graphene.set_hop(-1.0, 1, 0, [ 0, 1])

# solve model on path in k-space
path=[[0.0, 0.0],[0,1]]

kmesh=zeros((10,10),dtype=float)

kmesh=


kpts=k_path(path, 150)
print kpts
evals=graphene.solve_all(kpts)

# plot bandstructure
import matplotlib.pyplot as plt
plt.plot(evals[0, :])
plt.plot(evals[1, :])
plt.savefig("band.png")
