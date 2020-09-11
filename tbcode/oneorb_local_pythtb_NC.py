from __future__ import print_function

import numpy as np # numerics for matrices
import sys # for exiting
import copy # for deepcopying

class tb_model(object):
    def __init__(self,dim_k,dim_r,lat=None,orb=None,per=None,nspin=1):

        # initialize _dim_k = dimensionality of k-space (integer)
        if not _is_int(dim_k):
            raise Exception("\n\nArgument dim_k not an integer")
        if dim_k < 0 or dim_k > 4:
            raise Exception("\n\nArgument dim_k out of range. Must be between 0 and 4.")
        self._dim_k=dim_k

        # initialize _dim_r = dimensionality of r-space (integer)
        if not _is_int(dim_r):
            raise Exception("\n\nArgument dim_r not an integer")
        if dim_r < dim_k or dim_r > 4:
            raise Exception("\n\nArgument dim_r out of range. Must be dim_r>=dim_k and dim_r<=4.")
        self._dim_r=dim_r

        # initialize _lat = lattice vectors, array of dim_r*dim_r
        #   format is _lat(lat_vec_index,cartesian_index)
        # special option: 'unit' implies unit matrix, also default value
        if lat is 'unit' or lat is None:
            self._lat=np.identity(dim_r,float)
            print(" Lattice vectors not specified! I will use identity matrix.")
        elif type(lat).__name__ not in ['list','ndarray']:
            raise Exception("\n\nArgument lat is not a list.")
        else:
            self._lat=np.array(lat,dtype=float)
            if self._lat.shape!=(dim_r,dim_r):
                raise Exception("\n\nWrong lat array dimensions")
        # check that volume is not zero and that have right handed system
        if dim_r>0:
            if np.abs(np.linalg.det(self._lat))<1.0E-6:
                raise Exception("\n\nLattice vectors length/area/volume too close to zero, or zero.")
            if np.linalg.det(self._lat)<0.0:
                raise Exception("\n\nLattice vectors need to form right handed system.")

        # initialize _norb = number of basis orbitals per cell
        #   and       _orb = orbital locations, in reduced coordinates
        #   format is _orb(orb_index,lat_vec_index)
        # special option: 'bravais' implies one atom at origin
        if orb is 'bravais' or orb is None:
            self._norb=1
            self._orb=np.zeros((1,dim_r))
            print(" Orbital positions not specified. I will assume a single orbital at the origin.")
        elif _is_int(orb):
            self._norb=orb
            self._orb=np.zeros((orb,dim_r))
            print(" Orbital positions not specified. I will assume ",orb," orbitals at the origin")
        elif type(orb).__name__ not in ['list','ndarray']:
            raise Exception("\n\nArgument orb is not a list or an integer")
        else:
            self._orb=np.array(orb,dtype=float)
            if len(self._orb.shape)!=2:
                raise Exception("\n\nWrong orb array rank")
            self._norb=self._orb.shape[0] # number of orbitals
            if self._orb.shape[1]!=dim_r:
                raise Exception("\n\nWrong orb array dimensions")

        # choose which self._dim_k out of self._dim_r dimensions are
        # to be considered periodic.        
        if per==None:
            # by default first _dim_k dimensions are periodic
            self._per=list(range(self._dim_k))
        else:
            if len(per)!=self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per=per

        # remember number of spin components
        if nspin not in [1,2]:
            raise Exception("\n\nWrong value of nspin, must be 1 or 2!")
        self._nspin=nspin

        # by default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal=True

        # compute number of electronic states at each k-point
        self._nsta=self._norb*self._nspin
        
        # Initialize onsite energies to zero
        if self._nspin==1:
            self._site_energies=np.zeros((self._norb),dtype=float)
        elif self._nspin==2:
            self._site_energies=np.zeros((self._norb,2,2),dtype=complex)
        # remember which onsite energies user has specified
        self._site_energies_specified=np.zeros(self._norb,dtype=bool)
        self._site_energies_specified[:]=False
        
        # Initialize hoppings to empty list
        self._hoppings=[]

        # The onsite energies and hoppings are not specified
        # when creating a 'tb_model' object.  They are speficied
        # subsequently by separate function calls defined below.

    def set_onsite(self,onsite_en,ind_i=None,mode="set"):
        if ind_i==None:
            if (len(onsite_en)!=self._norb):
                raise Exception("\n\nWrong number of site energies")
        # make sure ind_i is not out of scope
        if ind_i!=None:
            if ind_i<0 or ind_i>=self._norb:
                raise Exception("\n\nIndex ind_i out of scope.")
        # make sure that onsite terms are real/hermitian
        if ind_i!=None:
            to_check=[onsite_en]
        else:
            to_check=onsite_en
        for ons in to_check:
            if np.array(ons).shape==():
                if np.abs(np.array(ons)-np.array(ons).conjugate())>1.0E-8:
                    raise Exception("\n\nOnsite energy should not have imaginary part!")
            elif np.array(ons).shape==(4,):
                if np.max(np.abs(np.array(ons)-np.array(ons).conjugate()))>1.0E-8:
                    raise Exception("\n\nOnsite energy or Zeeman field should not have imaginary part!")
            elif np.array(ons).shape==(2,2):
                if np.max(np.abs(np.array(ons)-np.array(ons).T.conjugate()))>1.0E-8:
                    raise Exception("\n\nOnsite matrix should be Hermitian!")
        # specifying onsite energies from scratch, can be called only once
        if mode.lower()=="set":
            # specifying only one site at a time
            if ind_i!=None:
                # make sure we specify things only once
                if self._site_energies_specified[ind_i]==True:
                    raise Exception("\n\nOnsite energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                    self._site_energies[ind_i]=self._val_to_block(onsite_en)
                    self._site_energies_specified[ind_i]=True
            # specifying all sites at once
            else:
                # make sure we specify things only once
                if True in self._site_energies_specified[ind_i]:
                    raise Exception("\n\nSome or all onsite energies were already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                    for i in range(self._norb):
                        self._site_energies[i]=self._val_to_block(onsite_en[i])
                    self._site_energies_specified[:]=True
        # reset values of onsite terms, without adding to previous value
        elif mode.lower()=="reset":
            # specifying only one site at a time
            if ind_i!=None:
                self._site_energies[ind_i]=self._val_to_block(onsite_en)
                self._site_energies_specified[ind_i]=True
            # specifying all sites at once
            else:
                for i in range(self._norb):
                    self._site_energies[i]=self._val_to_block(onsite_en[i])
                self._site_energies_specified[:]=True
        # add to previous value
        elif mode.lower()=="add":
            # specifying only one site at a time
            if ind_i!=None:
                self._site_energies[ind_i]+=self._val_to_block(onsite_en)
                self._site_energies_specified[ind_i]=True
            # specifying all sites at once
            else:
                for i in range(self._norb):
                    self._site_energies[i]+=self._val_to_block(onsite_en[i])
                self._site_energies_specified[:]=True
        else:
            raise Exception("\n\nWrong value of mode parameter")
        
    def set_hop(self,hop_amp,ind_i,ind_j,ind_R=None,mode="set",allow_conjugate_pair=False):
        #
        if self._dim_k!=0 and (ind_R is None):
            raise Exception("\n\nNeed to specify ind_R!")
        # if necessary convert from integer to array
        if self._dim_k==1 and _is_int(ind_R):
            tmpR=np.zeros(self._dim_r,dtype=int)
            tmpR[self._per]=ind_R
            ind_R=tmpR
        # check length of ind_R
        if self._dim_k!=0:
            if len(ind_R)!=self._dim_r:
                raise Exception("\n\nLength of input ind_R vector must equal dim_r! Even if dim_k<dim_r.")
        # make sure ind_i and ind_j are not out of scope
        if ind_i<0 or ind_i>=self._norb:
            raise Exception("\n\nIndex ind_i out of scope.")
        if ind_j<0 or ind_j>=self._norb:
            raise Exception("\n\nIndex ind_j out of scope.")        
        # do not allow onsite hoppings to be specified here because then they
        # will be double-counted
        if self._dim_k==0:
            if ind_i==ind_j:
                raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        else:
            if ind_i==ind_j:
                all_zer=True
                for k in self._per:
                    if int(ind_R[k])!=0:
                        all_zer=False
                if all_zer==True:
                    raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
        #
        # make sure that if <i|H|j+R> is specified that <j|H|i-R> is not!
        if allow_conjugate_pair==False:
            for h in self._hoppings:
                if ind_i==h[2] and ind_j==h[1]:
                    if self._dim_k==0:
                        raise Exception(\
"""\n
Following matrix element was already implicitely specified:
   i="""+str(ind_i)+" j="+str(ind_j)+"""
Remember, specifying <i|H|j> automatically specifies <j|H|i>.  For
consistency, specify all hoppings for a given bond in the same
direction.  (Or, alternatively, see the documentation on the
'allow_conjugate_pair' flag.)
""")
                    elif False not in (np.array(ind_R)[self._per]==(-1)*np.array(h[3])[self._per]):
                        raise Exception(\
"""\n
Following matrix element was already implicitely specified:
   i="""+str(ind_i)+" j="+str(ind_j)+" R="+str(ind_R)+"""
Remember,specifying <i|H|j+R> automatically specifies <j|H|i-R>.  For
consistency, specify all hoppings for a given bond in the same
direction.  (Or, alternatively, see the documentation on the
'allow_conjugate_pair' flag.)
""")
        # convert to 2by2 matrix if needed
        hop_use=self._val_to_block(hop_amp)
        # hopping term parameters to be stored
        if self._dim_k==0:
            new_hop=[hop_use,int(ind_i),int(ind_j)]
        else:
            new_hop=[hop_use,int(ind_i),int(ind_j),np.array(ind_R)]
        #
        # see if there is a hopping term with same i,j,R
        use_index=None
        for iih,h in enumerate(self._hoppings):
            # check if the same
            same_ijR=False 
            if ind_i==h[1] and ind_j==h[2]:
                if self._dim_k==0:
                    same_ijR=True
                else:
                    if False not in (np.array(ind_R)[self._per]==np.array(h[3])[self._per]):
                        same_ijR=True
            # if they are the same then store index of site at which they are the same
            if same_ijR==True:
                use_index=iih
        #
        # specifying hopping terms from scratch, can be called only once
        if mode.lower()=="set":
            # make sure we specify things only once
            if use_index!=None:
                raise Exception("\n\nHopping energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
            else:
                self._hoppings.append(new_hop)
        # reset value of hopping term, without adding to previous value
        elif mode.lower()=="reset":
            if use_index!=None:
                self._hoppings[use_index]=new_hop
            else:
                self._hoppings.append(new_hop)
        # add to previous value
        elif mode.lower()=="add":
            if use_index!=None:
                self._hoppings[use_index][0]+=new_hop[0]
            else:
                self._hoppings.append(new_hop)
        else:
            raise Exception("\n\nWrong value of mode parameter")

    def _val_to_block(self,val):
        # spinless case
        if self._nspin==1:
            return val
        # spinfull case
        elif self._nspin==2:
            # matrix to return
            ret=np.zeros((2,2),dtype=complex)
            # 
            use_val=np.array(val)
            # only one number is given
            if use_val.shape==():
                ret[0,0]+=use_val
                ret[1,1]+=use_val
            # if four numbers are given
            elif use_val.shape==(4,):
                # diagonal
                ret[0,0]+=use_val[0]
                ret[1,1]+=use_val[0]
                # sigma_x
                ret[0,1]+=use_val[1]
                ret[1,0]+=use_val[1]                # sigma_y
                ret[0,1]+=use_val[2]*(-1.0j)
                ret[1,0]+=use_val[2]*( 1.0j)
                # sigma_z
                ret[0,0]+=use_val[3]
                ret[1,1]+=use_val[3]*(-1.0)        
            # if 2 by 2 matrix is given
            elif use_val.shape==(2,2):
                return use_val
            else:
                raise Exception(\
"""\n
Wrong format of the on-site or hopping term. Must be single number, or
in the case of a spinfull model can be array of four numbers or 2x2
matrix.""")            
            return ret        
        
    def display(self):
        r"""
        Prints on the screen some information about this tight-binding
        model. This function doesn't take any parameters.
        """
        print('---------------------------------------')
        print('report of tight-binding model')
        print('---------------------------------------')
        print('k-space dimension           =',self._dim_k)
        print('r-space dimension           =',self._dim_r)
        print('number of spin components   =',self._nspin)
        print('periodic directions         =',self._per)
        print('number of orbitals          =',self._norb)
        print('number of electronic states =',self._nsta)
        print('lattice vectors:')
        for i,o in enumerate(self._lat):
            print(" #",_nice_int(i,2)," ===>  [", end=' ')
            for j,v in enumerate(o):
                print(_nice_float(v,7,4), end=' ')
                if j!=len(o)-1:
                    print(",", end=' ')
            print("]")
        print('positions of orbitals:')
        for i,o in enumerate(self._orb):
            print(" #",_nice_int(i,2)," ===>  [", end=' ')
            for j,v in enumerate(o):
                print(_nice_float(v,7,4), end=' ')
                if j!=len(o)-1:
                    print(",", end=' ')
            print("]")
        print('site energies:')
        for i,site in enumerate(self._site_energies):
            print(" #",_nice_int(i,2)," ===>  ", end=' ')
            if self._nspin==1:
                print(_nice_float(site,7,4))
            elif self._nspin==2:
                print(str(site).replace("\n"," "))
        print('hoppings:')
        for i,hopping in enumerate(self._hoppings):
            print("<",_nice_int(hopping[1],2),"| H |",_nice_int(hopping[2],2), end=' ')
            if len(hopping)==4:
                print("+ [", end=' ')
                for j,v in enumerate(hopping[3]):
                    print(_nice_int(v,2), end=' ')
                    if j!=len(hopping[3])-1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(">     ===> ", end=' ')
            if self._nspin==1:
                print(_nice_complex(hopping[0],7,4))
            elif self._nspin==2:
                print(str(hopping[0]).replace("\n"," "))
        print('hopping distances:')
        for i,hopping in enumerate(self._hoppings):
            print("|  pos(",_nice_int(hopping[1],2),")  - pos(",_nice_int(hopping[2],2), end=' ')
            if len(hopping)==4:
                print("+ [", end=' ')
                for j,v in enumerate(hopping[3]):
                    print(_nice_int(v,2), end=' ')
                    if j!=len(hopping[3])-1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(") |  =  ", end=' ')
            pos_i=np.dot(self._orb[hopping[1]],self._lat)
            pos_j=np.dot(self._orb[hopping[2]],self._lat)
            if len(hopping)==4:
                pos_j+=np.dot(hopping[3],self._lat)
            dist=np.linalg.norm(pos_j-pos_i)
            print (_nice_float(dist,7,4))

        print()


    def _gen_ham(self,k_input=None):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt=np.array(k_input)
        if not (k_input is None):
            # if kpnt is just a number then convert it to an array
            if len(kpnt.shape)==0:
                kpnt=np.array([kpnt])
            # check that k-vector is of corect size
            if kpnt.shape!=(self._dim_k,):
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self._dim_k!=0:
                raise Exception("\n\nHave to provide a k-vector!")
        # zero the Hamiltonian matrix
        if self._nspin==1:
            ham=np.zeros((self._norb,self._norb),dtype=complex)
        elif self._nspin==2:
            ham=np.zeros((self._norb,2,self._norb,2),dtype=complex)
        # modify diagonal elements
        for i in range(self._norb):
            if self._nspin==1:
                ham[i,i]=self._site_energies[i]
            elif self._nspin==2:
                ham[i,:,i,:]=self._site_energies[i]
        # go over all hoppings
        for hopping in self._hoppings:
            # get all data for the hopping parameter
            if self._nspin==1:
                amp=complex(hopping[0])
            elif self._nspin==2:
                amp=np.array(hopping[0],dtype=complex)
            i=hopping[1]
            j=hopping[2]
            # in 0-dim case there is no phase factor
            if self._dim_k>0:
                ind_R=np.array(hopping[3],dtype=float)
                # vector from one site to another
                rv=-self._orb[i,:]+self._orb[j,:]+ind_R
                # Take only components of vector which are periodic
                rv=rv[self._per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase=np.exp((2.0j)*np.pi*np.dot(kpnt,rv))
                amp=amp*phase
            # add this hopping into a matrix and also its conjugate
            if self._nspin==1:
                ham[i,j]+=amp
                ham[j,i]+=amp.conjugate()
            elif self._nspin==2:
                ham[i,:,j,:]+=amp
                ham[j,:,i,:]+=amp.T.conjugate()
        print ('SHAPE:', ham.shape)
        print ('Hamiltonian:', ham.real)
        return ham

    def _gen_ham2(self,k_input, t=1.):
        kpnt=2*np.pi*np.array(k_input)
        ham=np.zeros((self._norb,self._norb),dtype=complex)
    
        for i in range(self._norb):
            for j in range(self._norb):
                ham[i,j] = -2*t*(np.cos(kpnt[0])+np.cos(kpnt[1]))

        print (ham.shape)
        return ham

    def _sol_ham(self,ham,eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        if self._nspin==1:
            ham_use=ham
        elif self._nspin==2:
            ham_use=ham.reshape((2*self._norb,2*self._norb))
        # check that matrix is hermitian
        if np.max(ham_use-ham_use.T.conj())>1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if eig_vectors==False: # only find eigenvalues
            eval=np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval=_nicefy_eig(eval)
            return np.array(eval,dtype=float)
        else: # find eigenvalues and eigenvectors
            (eval,eig)=np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig=eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval,eig)=_nicefy_eig(eval,eig)
            # reshape eigenvectors if doing a spinfull calculation
            if self._nspin==2:
                eig=eig.reshape((self._nsta,self._norb,2))
            return (eval,eig)

    def solve_all(self,k_list=None,eig_vectors=False):
        # if not 0-dim case
        if not (k_list is None):
            nkp=len(k_list) # number of k points
            # first initialize matrices for all return data
            #    indices are [band,kpoint]
            ret_eval=np.zeros((self._nsta,nkp),dtype=float)
            #    indices are [band,kpoint,orbital,spin]
            if self._nspin==1:
                ret_evec=np.zeros((self._nsta,nkp,self._norb),dtype=complex)
            elif self._nspin==2:
                ret_evec=np.zeros((self._nsta,nkp,self._norb,2),dtype=complex)
            # go over all kpoints
            for i,k in enumerate(k_list):
                # generate Hamiltonian at that point
                ham=self._gen_ham2(k)
                # solve Hamiltonian
                if eig_vectors==False:
                    eval=self._sol_ham(ham,eig_vectors=eig_vectors)
                    ret_eval[:,i]=eval[:]
                else:
                    (eval,evec)=self._sol_ham(ham,eig_vectors=eig_vectors)
                    ret_eval[:,i]=eval[:]
                    if self._nspin==1:
                        ret_evec[:,i,:]=evec[:,:]
                    elif self._nspin==2:
                        ret_evec[:,i,:,:]=evec[:,:,:]
            # return stuff
            if eig_vectors==False:
                # indices of eval are [band,kpoint]
                return ret_eval
            else:
                # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
                return (ret_eval,ret_evec)
        else: # 0 dim case
            # generate Hamiltonian
            ham=self._gen_ham()
            # solve
            if eig_vectors==False:
                eval=self._sol_ham(ham,eig_vectors=eig_vectors)
                # indices of eval are [band]
                return eval
            else:
                (eval,evec)=self._sol_ham(ham,eig_vectors=eig_vectors)
                # indices of eval are [band] and of evec are [band,orbital,spin]
                return (eval,evec)

    def solve_one(self,k_point=None,eig_vectors=False):
        r"""

        Similar to :func:`pythtb.tb_model.solve_all` but solves tight-binding
        model for only one k-vector.

        """
        # if not 0-dim case
        if not (k_point is None):
            if eig_vectors==False:
                eval=self.solve_all([k_point],eig_vectors=eig_vectors)
                # indices of eval are [band]
                return eval[:,0]
            else:
                (eval,evec)=self.solve_all([k_point],eig_vectors=eig_vectors)
                # indices of eval are [band] for evec are [band,orbital,spin]
                if self._nspin==1:
                    return (eval[:,0],evec[:,0,:])
                elif self._nspin==2:
                    return (eval[:,0],evec[:,0,:,:])
        else:
            # do the same as solve_all
            return self.solve_all(eig_vectors=eig_vectors)

    def cut_piece(self,num,fin_dir,glue_edgs=False):
        if self._dim_k ==0:
            raise Exception("\n\nModel is already finite")
        if not _is_int(num):
            raise Exception("\n\nArgument num not an integer")

        # check value of num
        if num<1:
            raise Exception("\n\nArgument num must be positive!")
        if num==1 and glue_edgs==True:
            raise Exception("\n\nCan't have num==1 and glueing of the edges!")

        # generate orbitals of a finite model
        fin_orb=[]
        onsite=[] # store also onsite energies
        for i in range(num): # go over all cells in finite direction
            for j in range(self._norb): # go over all orbitals in one cell
                # make a copy of j-th orbital
                orb_tmp=np.copy(self._orb[j,:])
                # change coordinate along finite direction
                orb_tmp[fin_dir]+=float(i)
                # add to the list
                fin_orb.append(orb_tmp)
                # do the onsite energies at the same time
                onsite.append(self._site_energies[j])
        onsite=np.array(onsite)
        fin_orb=np.array(fin_orb)

        # generate periodic directions of a finite model
        fin_per=copy.deepcopy(self._per)
        # find if list of periodic directions contains the one you
        # want to make finite
        if fin_per.count(fin_dir)!=1:
            raise Exception("\n\nCan not make model finite along this direction!")
        # remove index which is no longer periodic
        fin_per.remove(fin_dir)

        # generate object of tb_model type that will correspond to a cutout
        fin_model=tb_model(self._dim_k-1,
                           self._dim_r,
                           copy.deepcopy(self._lat),
                           fin_orb,
                           fin_per,
                           self._nspin)

        # remember if came from w90
        fin_model._assume_position_operator_diagonal=self._assume_position_operator_diagonal

        # now put all onsite terms for the finite model
        fin_model.set_onsite(onsite,mode="reset")

        # put all hopping terms
        for c in range(num): # go over all cells in finite direction
            for h in range(len(self._hoppings)): # go over all hoppings in one cell
                # amplitude of the hop is the same
                amp=self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R=copy.deepcopy(self._hoppings[h][3])
                jump_fin=ind_R[fin_dir] # store by how many cells is the hopping in finite direction
                if fin_model._dim_k!=0:
                    ind_R[fin_dir]=0 # one of the directions now becomes finite

                # index of "from" and "to" hopping indices
                hi=self._hoppings[h][1] + c*self._norb
                #   have to compensate  for the fact that ind_R in finite direction
                #   will not be used in the finite model
                hj=self._hoppings[h][2] + (c + jump_fin)*self._norb

                # decide whether this hopping should be added or not
                to_add=True
                # if edges are not glued then neglect all jumps that spill out
                if glue_edgs==False:
                    if hj<0 or hj>=self._norb*num:
                        to_add=False
                # if edges are glued then do mod division to wrap up the hopping
                else:
                    hj=int(hj)%int(self._norb*num)

                # add hopping to a finite model
                if to_add==True:
                    if fin_model._dim_k==0:
                        fin_model.set_hop(amp,hi,hj,mode="add",allow_conjugate_pair=True)
                    else:
                        fin_model.set_hop(amp,hi,hj,ind_R,mode="add",allow_conjugate_pair=True)

        return fin_model

    def reduce_dim(self,remove_k,value_k):
        #
        if self._dim_k==0:
            raise Exception("\n\nCan not reduce dimensionality even further!")
        # make a copy
        red_tb=copy.deepcopy(self)
        # make one of the directions not periodic
        red_tb._per.remove(remove_k)
        red_tb._dim_k=len(red_tb._per)
        # check that really removed one and only one direction
        if red_tb._dim_k!=self._dim_k-1:
            raise Exception("\n\nSpecified wrong dimension to reduce!")
        
        # specify hopping terms from scratch
        red_tb._hoppings=[]
        # set all hopping parameters for this value of value_k
        for h in range(len(self._hoppings)):
            hop=self._hoppings[h]
            if self._nspin==1:
                amp=complex(hop[0])
            elif self._nspin==2:
                amp=np.array(hop[0],dtype=complex)
            i=hop[1]; j=hop[2]
            ind_R=np.array(hop[3],dtype=int)
            # vector from one site to another
            rv=-red_tb._orb[i,:]+red_tb._orb[j,:]+np.array(ind_R,dtype=float)
            # take only r-vector component along direction you are not making periodic
            rv=rv[remove_k]
            # Calculate the part of hopping phase, only for this direction
            phase=np.exp((2.0j)*np.pi*(value_k*rv))
            # store modified version of the hop
            # Since we are getting rid of one dimension, it could be that now
            # one of the hopping terms became onsite term because one direction
            # is no longer periodic
            if i==j and (False not in (np.array(ind_R[red_tb._per],dtype=int)==0)):
                if ind_R[remove_k]==0:
                    # in this case this is really an onsite term
                    red_tb.set_onsite(amp*phase,i,mode="add")
                else:
                    # in this case must treat both R and -R because that term would
                    # have been counted twice without dimensional reduction
                    if self._nspin==1:
                        red_tb.set_onsite(amp*phase+(amp*phase).conj(),i,mode="add")
                    elif self._nspin==2:
                        red_tb.set_onsite(amp*phase+(amp.T*phase).conj(),i,mode="add")
            else:
                # just in case make the R vector zero along the reduction dimension
                ind_R[remove_k]=0
                # add hopping term
                red_tb.set_hop(amp*phase,i,j,ind_R,mode="add",allow_conjugate_pair=True)
                
        return red_tb


    def _shift_to_home(self):
        """Shifts all orbital positions to the home unit cell. After
        this function is called all reduced coordiantes of orbitals
        will be between 0 and 1. It may be useful to call this
        function after using make_supercell."""
        
        # go over all orbitals
        for i in range(self._norb):
            cur_orb=self._orb[i]
            # compute orbital in the home cell
            round_orb=(np.array(cur_orb)+1.0E-6)%1.0
            # find displacement vector needed to bring back to home cell
            disp_vec=np.array(np.round(cur_orb-round_orb),dtype=int)
            # check if have at least one non-zero component
            if True in (disp_vec!=0):
                # shift orbital
                self._orb[i]-=np.array(disp_vec,dtype=float)
                # shift also hoppings
                if self._dim_k!=0:
                    for h in range(len(self._hoppings)):
                        if self._hoppings[h][1]==i:
                            self._hoppings[h][3]-=disp_vec
                        if self._hoppings[h][2]==i:
                            self._hoppings[h][3]+=disp_vec


    def remove_orb(self,to_remove):

        # if a single integer is given, convert to a list with one element
        if _is_int(to_remove):
            orb_index=[to_remove]
        else:
            orb_index=copy.deepcopy(to_remove)

        # check range of indices
        for i,orb_ind in enumerate(orb_index):
            if orb_ind < 0 or orb_ind > self._norb-1 or (not _is_int(orb_ind)):
                raise Exception("\n\nSpecified wrong orbitals to remove!")
        for i,ind1 in enumerate(orb_index):
            for ind2 in orb_index[i+1:]:
                if ind1==ind2:
                    raise Exception("\n\nSpecified duplicate orbitals to remove!")

        # put the orbitals to be removed in desceding order
        orb_index = sorted(orb_index,reverse=True)

        # make copy of a model
        ret=copy.deepcopy(self)

        # adjust some variables in the new model
        ret._norb-=len(orb_index)
        ret._nsta-=len(orb_index)*self._nspin
        # remove indices one by one
        for i,orb_ind in enumerate(orb_index):
            # adjust variables
            ret._orb = np.delete(ret._orb,orb_ind,0)
            ret._site_energies = np.delete(ret._site_energies,orb_ind,0)
            ret._site_energies_specified = np.delete(ret._site_energies_specified,orb_ind)
            # adjust hopping terms (in reverse)
            for j in range(len(ret._hoppings)-1,-1,-1):
                h=ret._hoppings[j]
                # remove all terms that involve this orbital
                if h[1]==orb_ind or h[2]==orb_ind:
                    del ret._hoppings[j]
                else: # otherwise modify term
                    if h[1]>orb_ind:
                        ret._hoppings[j][1]-=1
                    if h[2]>orb_ind:
                        ret._hoppings[j][2]-=1
        return ret

    def k_path(self,kpts,nk,report=True):
    
        # processing of special cases for kpts
        if kpts=='full':
            # full Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5],[1.]])
        elif kpts=='fullc':
            # centered full Brillouin zone for 1D case
            k_list=np.array([[-0.5],[0.],[0.5]])
        elif kpts=='half':
            # half Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5]])
        else:
            k_list=np.array(kpts)
    
        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape)==1 and self._dim_k==1:
            k_list=np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1]!=self._dim_k:
            print('input k-space dimension is',k_list.shape[1])
            print('k-space dimension taken from model is',self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk<k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes=k_list.shape[0]
    
        # extract the lattice vectors from the TB model
        lat_per=np.copy(self._lat)
        # choose only those that correspond to periodic directions
        lat_per=lat_per[self._per]    
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per,lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node=np.zeros(n_nodes,dtype=float)
        for n in range(1,n_nodes):
            dk = k_list[n]-k_list[n-1]
            dklen = np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
            k_node[n]=k_node[n-1]+dklen
    
        # Find indices of nodes in interpolated list
        node_index=[0]
        for n in range(1,n_nodes-1):
            frac=k_node[n]/k_node[-1]
            node_index.append(int(round(frac*(nk-1))))
        node_index.append(nk-1)
    
        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist=np.zeros(nk,dtype=float)
        #   array listing the interpolated k-points    
        k_vec=np.zeros((nk,self._dim_k),dtype=float)
    
        # go over all kpoints
        k_vec[0]=k_list[0]
        for n in range(1,n_nodes):
            n_i=node_index[n-1]
            n_f=node_index[n]
            kd_i=k_node[n-1]
            kd_f=k_node[n]
            k_i=k_list[n-1]
            k_f=k_list[n]
            for j in range(n_i,n_f+1):
                frac=float(j-n_i)/float(n_f-n_i)
                k_dist[j]=kd_i+frac*(kd_f-kd_i)
                k_vec[j]=k_i+frac*(k_f-k_i)

        return (k_vec,k_dist,k_node)

    def ignore_position_operator_offdiagonal(self):
        """Call to this function enables one to approximately compute
        Berry-like objects from tight-binding models that were
        obtained from Wannier90."""  
        self._assume_position_operator_diagonal=True

    def position_matrix(self, evec, dir):

        # make sure specified direction is not periodic!
        if dir in self._per:
            raise Exception("Can not compute position matrix elements along periodic direction!")
        # make sure direction is not out of range
        if dir<0 or dir>=self._dim_r:
            raise Exception("Direction out of range!")
        
        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # get coordinates of orbitals along the specified direction
        pos_tmp=self._orb[:,dir]
        # reshape arrays in the case of spinfull calculation
        if self._nspin==2:
            # tile along spin direction if needed
            pos_use=np.tile(pos_tmp,(2,1)).transpose().flatten()
            # also flatten the state along the spin index
            evec_use=evec.reshape((evec.shape[0],evec.shape[1]*evec.shape[2]))                
        else:
            pos_use=pos_tmp
            evec_use=evec

        # position matrix elements
        pos_mat=np.zeros((evec_use.shape[0],evec_use.shape[0]),dtype=complex)
        # go over all bands
        for i in range(evec_use.shape[0]):
            for j in range(evec_use.shape[0]):
                pos_mat[i,j]=np.dot(evec_use[i].conj(),pos_use*evec_use[j])

        # make sure matrix is hermitian
        if np.max(pos_mat-pos_mat.T.conj())>1.0E-9:
            raise Exception("\n\n Position matrix is not hermitian?!")

        return pos_mat

    def position_expectation(self,evec,dir):
        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        pos_exp=self.position_matrix(evec,dir).diagonal()
        return np.array(np.real(pos_exp),dtype=float)

    def position_hwf(self,evec,dir,hwf_evec=False,basis="orbital"):
        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # get position matrix
        pos_mat=self.position_matrix(evec,dir)

        # diagonalize
        if hwf_evec==False:
            hwfc=np.linalg.eigvalsh(pos_mat)
            # sort eigenvalues and convert to real numbers
            hwfc=_nicefy_eig(hwfc)
            return np.array(hwfc,dtype=float)
        else: # find eigenvalues and eigenvectors
            (hwfc,hwf)=np.linalg.eigh(pos_mat)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            hwf=hwf.T
            # sort evectors, eigenvalues and convert to real numbers
            (hwfc,hwf)=_nicefy_eig(hwfc,hwf)
            # convert to right basis
            if basis.lower().strip()=="bloch":
                return (hwfc,hwf)
            elif basis.lower().strip()=="orbital":
                if self._nspin==1:
                    ret_hwf=np.zeros((hwf.shape[0],self._norb),dtype=complex)
                    # sum over bloch states to get hwf in orbital basis
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i]=np.dot(hwf[i],evec)
                    hwf=ret_hwf
                else:
                    ret_hwf=np.zeros((hwf.shape[0],self._norb*2),dtype=complex)
                    # get rid of spin indices
                    evec_use=evec.reshape([hwf.shape[0],self._norb*2])
                    # sum over states
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i]=np.dot(hwf[i],evec_use)
                    # restore spin indices
                    hwf=ret_hwf.reshape([hwf.shape[0],self._norb,2])
                return (hwfc,hwf)
            else:
                raise Exception("\n\nBasis must be either bloch or orbital!")


# keeping old name for backwards compatibility
# will be removed in future
tb_model.set_sites=tb_model.set_onsite
tb_model.add_hop=tb_model.set_hop
tbmodel=tb_model

class wf_array(object):
    def __init__(self,model,mesh_arr):
        # number of electronic states for each k-point
        self._nsta=model._nsta
        # number of spin components
        self._nspin=model._nspin
        # number of orbitals
        self._norb=model._norb
        # store orbitals from the model
        self._orb=np.copy(model._orb)
        # store entire model as well
        self._model=copy.deepcopy(model)
        # store dimension of array of points on which to keep wavefunctions
        self._mesh_arr=np.array(mesh_arr)        
        self._dim_arr=len(self._mesh_arr)
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_arr<=1).tolist():
            raise Exception("\n\nDimension of wf_array object in each direction must be 2 or larger.")
        # generate temporary array used later to generate object ._wfs
        wfs_dim=np.copy(self._mesh_arr)
        wfs_dim=np.append(wfs_dim,self._nsta)
        wfs_dim=np.append(wfs_dim,self._norb)
        if self._nspin==2:
            wfs_dim=np.append(wfs_dim,self._nspin)            
        # store wavefunctions here in the form _wfs[kx_index,ky_index, ... ,band,orb,spin]
        self._wfs=np.zeros(wfs_dim,dtype=complex)

    def solve_on_grid(self,start_k):
        # check dimensionality
        if self._dim_arr!=self._model._dim_k:
            raise Exception("\n\nIf using solve_on_grid method, dimension of wf_array must equal dim_k of the tight-binding model!")
        # to return gaps at all k-points
        if self._nsta<=1:
            all_gaps=None # trivial case since there is only one band
        else:
            gap_dim=np.copy(self._mesh_arr)-1
            gap_dim=np.append(gap_dim,self._nsta-1)
            all_gaps=np.zeros(gap_dim,dtype=float)
        #
        if self._dim_arr==1:
            # don't need to go over the last point because that will be
            # computed in the impose_pbc call
            for i in range(self._mesh_arr[0]-1):
                # generate a kpoint
                kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1)]
                # solve at that point
                (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
                # store wavefunctions
                self[i]=evec
                # store gaps
                if all_gaps is not None:
                    all_gaps[i,:]=eval[1:]-eval[:-1]
            # impose boundary conditions
            self.impose_pbc(0,self._model._per[0])
        elif self._dim_arr==2:
            for i in range(self._mesh_arr[0]-1):
                for j in range(self._mesh_arr[1]-1):
                    kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1),\
                         start_k[1]+float(j)/float(self._mesh_arr[1]-1)]
                    (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
                    self[i,j]=evec
                    if all_gaps is not None:
                        all_gaps[i,j,:]=eval[1:]-eval[:-1]
            for dir in range(2):
                self.impose_pbc(dir,self._model._per[dir])
        elif self._dim_arr==3:
            for i in range(self._mesh_arr[0]-1):
                for j in range(self._mesh_arr[1]-1):
                    for k in range(self._mesh_arr[2]-1):
                        kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1),\
                             start_k[1]+float(j)/float(self._mesh_arr[1]-1),\
                             start_k[2]+float(k)/float(self._mesh_arr[2]-1)]
                        (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
                        self[i,j,k]=evec
                        if all_gaps is not None:
                            all_gaps[i,j,k,:]=eval[1:]-eval[:-1]
            for dir in range(3):
                self.impose_pbc(dir,self._model._per[dir])
        elif self._dim_arr==4:
            for i in range(self._mesh_arr[0]-1):
                for j in range(self._mesh_arr[1]-1):
                    for k in range(self._mesh_arr[2]-1):
                        for l in range(self._mesh_arr[3]-1):
                            kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1),\
                                     start_k[1]+float(j)/float(self._mesh_arr[1]-1),\
                                     start_k[2]+float(k)/float(self._mesh_arr[2]-1),\
                                     start_k[3]+float(l)/float(self._mesh_arr[3]-1)]
                            (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
                            self[i,j,k,l]=evec
                            if all_gaps is not None:
                                all_gaps[i,j,k,l,:]=eval[1:]-eval[:-1]
            for dir in range(4):
                self.impose_pbc(dir,self._model._per[dir])
        else:
            raise Exception("\n\nWrong dimensionality!")

        if all_gaps is not None:
            return all_gaps.min(axis=tuple(range(self._dim_arr)))
        else:
            return None
        
    def __check_key(self,key):
        # do some checks for 1D
        if self._dim_arr==1:
            if not _is_int(key):
                raise TypeError("Key should be an integer!")
            if key<(-1)*self._mesh_arr[0] or key>=self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do checks for higher dimension
        else:
            if len(key)!=self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i,k in enumerate(key):
                if not _is_int(k):
                    raise TypeError("Key should be set of integers!")
                if k<(-1)*self._mesh_arr[i] or k>=self._mesh_arr[i]:
                    raise IndexError("Key outside the range!")

    def __getitem__(self,key):
        # check that key is in the correct range
        self.__check_key(key)
        # return wavefunction
        return self._wfs[key]
    
    def __setitem__(self,key,value):
        # check that key is in the correct range
        self.__check_key(key)
        # store wavefunction
        self._wfs[key]=np.array(value,dtype=complex)

    def impose_pbc(self,mesh_dir,k_dir):
        if k_dir not in self._model._per:
            raise Exception("Periodic boundary condition can be specified only along periodic directions!")

        # Compute phase factors
        ffac=np.exp(-2.j*np.pi*self._orb[:,k_dir])
        if self._nspin==1:
            phase=ffac
        else:
            # for spinors, same phase multiplies both components
            phase=np.zeros((self._norb,2),dtype=complex)
            phase[:,0]=ffac
            phase[:,1]=ffac
        
        # Copy first eigenvector onto last one, multiplying by phase factors
        # We can use numpy broadcasting since the orbital index is last
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]*phase
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]*phase
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]*phase
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]*phase
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def impose_loop(self,mesh_dir):
        # Copy first eigenvector onto last one
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]
        else:
            raise Exception("\n\nWrong value of mesh_dir.")


    def berry_phase(self,occ,dir=None,contin=True,berry_evals=False):
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        #if dir<0 or dir>self._dim_arr-1:
        #  raise Exception("\n\nDirection key out of range")
        #
        # This could be coded more efficiently, but it is hard-coded for now.
        #
        # 1D case
        if self._dim_arr==1:
            # pick which wavefunctions to use
            wf_use=self._wfs[:,occ,:]
            # calculate berry phase
            ret=_one_berry_loop(wf_use,berry_evals)
        # 2D case
        elif self._dim_arr==2:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    wf_use=self._wfs[:,i,:,:][:,occ,:]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    wf_use=self._wfs[i,:,:,:][:,occ,:]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        # 3D case
        elif self._dim_arr==3:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[:,i,j,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[i,:,j,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==2:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[1]):
                        wf_use=self._wfs[i,j,:,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        else:
            raise Exception("\n\nWrong dimensionality!")

        # convert phases to numpy array
        if self._dim_arr>1 or berry_evals==True:
            ret=np.array(ret,dtype=float)

        # make phases of eigenvalues continuous
        if contin==True:
            # iron out 2pi jumps, make the gauge choice such that first phase in the
            # list is fixed, others are then made continuous.
            if berry_evals==False:
                # 2D case
                if self._dim_arr==2:
                    ret=_one_phase_cont(ret,ret[0])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0]
                        else: clos=ret[0,i-1]
                        ret[:,i]=_one_phase_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
            # make eigenvalues continuous. This does not take care of band-character
            # at band crossing for example it will just connect pairs that are closest
            # at neighboring points.
            else:
                # 2D case
                if self._dim_arr==2:
                    ret=_array_phases_cont(ret,ret[0,:])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0,:]
                        else: clos=ret[0,i-1,:]
                        ret[:,i]=_array_phases_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
        return ret

    def position_matrix(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_matrix`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_matrix(evec,dir)

    def position_expectation(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_expectation`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_expectation(evec,dir)

    def position_hwf(self, key, occ, dir, hwf_evec=False, basis="bloch"):
        """Similar to :func:`pythtb.tb_model.position_hwf`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_hwf(evec,dir,hwf_evec,basis)


    def berry_flux(self,occ,dirs=None,individual_phases=False):

        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # default case is to take first two directions for flux calculation
        if dirs==None:
            dirs=[0,1]

        # consistency checks
        if dirs[0]==dirs[1]:
            raise Exception("Need to specify two different directions for Berry flux calculation.")
        if dirs[0]>=self._dim_arr or dirs[1]>=self._dim_arr or dirs[0]<0 or dirs[1]<0:
            raise Exception("Direction for Berry flux calculation out of bounds.")

        # 2D case
        if self._dim_arr==2:
            # compute the fluxes through all plaquettes on the entire plane 
            ord=list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0]=dirs[0]
            ord[1]=dirs[1]
            plane_wfs=self._wfs.transpose(ord)
            # take bands of choice
            plane_wfs=plane_wfs[:,:,occ]

            # compute fluxes
            all_phases=_one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases==False:
                return all_phases.sum()
            else:
                return all_phases

        # 3D or 4D case
        elif self._dim_arr in [3,4]:
            # compute the fluxes through all plaquettes on the entire plane 
            ord=list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0]=dirs[0]
            ord[1]=dirs[1]

            # find directions over which we wish to loop
            ld=list(range(self._dim_arr))
            ld.remove(dirs[0])
            ld.remove(dirs[1])
            if len(ld)!=self._dim_arr-2:
                raise Exception("Hm, this should not happen? Inconsistency with the mesh size.")
            
            # add remaining indices
            if self._dim_arr==3:
                ord[2]=ld[0]
            if self._dim_arr==4:
                ord[2]=ld[0]
                ord[3]=ld[1]

            # reorder wavefunctions
            use_wfs=self._wfs.transpose(ord)

            # loop over the the remaining direction
            if self._dim_arr==3:
                slice_phases=np.zeros((self._mesh_arr[ord[2]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    # take a 2d slice
                    plane_wfs=use_wfs[:,:,i]
                    # take bands of choice
                    plane_wfs=plane_wfs[:,:,occ]
                    # compute fluxes on the slice
                    slice_phases[i,:,:]=_one_flux_plane(plane_wfs)
            elif self._dim_arr==4:
                slice_phases=np.zeros((self._mesh_arr[ord[2]],self._mesh_arr[ord[3]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    for j in range(self._mesh_arr[ord[3]]):
                        # take a 2d slice
                        plane_wfs=use_wfs[:,:,i,j]
                        # take bands of choice
                        plane_wfs=plane_wfs[:,:,occ]
                        # compute fluxes on the slice
                        slice_phases[i,j,:,:]=_one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases==False:
                return slice_phases.sum(axis=(-2,-1))
            else:
                return slice_phases

        else:
            raise Exception("\n\nWrong dimensionality!")


    def berry_curv(self,occ,individual_phases=False):
        r"""

      .. warning:: This function has been renamed as :func:`pythtb.berry_flux` and is provided
        here only for backwards compatibility with versions of pythtb prior to 1.7.0.  Please
        use related :func:`pythtb.berry_flux` as this function may not exist in future releases.

        """

        print(""" 

Warning:
  Usage of function berry_curv is discouraged.
  It has been renamed as berry_flux, which should be used instead.
""")
        return self.berry_flux(occ,individual_phases)



def k_path(kpts,nk,endpoint=True):
    r"""

      .. warning:: This function is here only for backwards compatibility 
        with version of pythtb prior to 1.7.0.  Please use related :func:`pythtb.tb_model.k_path`
        function as this function might not exist in the future releases of the code.

    """

    print(""" 

Warning:

  Usage of function k_path is discouraged.  
  Instead of the following code:
    k_vec=k_path(...)
  please use the following code:
    (k_vec,k_dist,k_node)=my_model.k_path(...)
  Note that this k_path function is a member of the tb_model class.

""")

    if kpts=='full':
        # this means the full Brillouin zone for 1D case
        if endpoint==True:
            return np.arange(nk+1,dtype=float)/float(nk)
        else:
            return np.arange(nk,dtype=float)/float(nk)
    elif kpts=='half':
        # this means the half Brillouin zone for 1D case
        if endpoint==True:
            return np.arange(nk+1,dtype=float)/float(2.*nk)
        else:
            return np.arange(nk,dtype=float)/float(2.*nk)
    else:
        # general case
        kint=[]
        k_list=np.array(kpts)
        # go over all kpoints
        for i in range(len(k_list)-1):
            # go over all steps
            for j in range(nk):
                cur=k_list[i]+(k_list[i+1]-k_list[i])*float(j)/float(nk)
                kint.append(cur)
        # add last point
        if endpoint==True:
            kint.append(k_list[-1])
        #
        kint=np.array(kint)
        return kint

def _nicefy_eig(eval,eig=None):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=np.array(eval.real,dtype=float)
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    if not (eig is None):
        eig=eig[args]
        return (eval,eig)
    return eval

# for nice justified printout
def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)
def _nice_int(x,just):
    return str(x).rjust(just)
def _nice_complex(x,just,rnd):
    ret=""
    ret+=_nice_float(complex(x).real,just,rnd)
    if complex(x).imag<0.0:
        ret+=" - "
    else:
        ret+=" + "
    ret+=_nice_float(abs(complex(x).imag),just,rnd)
    ret+=" i"
    return ret
    
def _wf_dpr(wf1,wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(),wf2.flatten())

def _one_berry_loop(wf,berry_evals=False):
    """Do one Berry phase calculation (also returns a product of M
    matrices).  Always returns numbers between -pi and pi.  wf has
    format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
    Assumes that first and last k-point are the same. Therefore if
    there are n wavefunctions in total, will calculate phase along n-1
    links only!  If berry_evals is True then will compute phases for
    individual states, these corresponds to 1d hybrid Wannier
    function centers. Otherwise just return one number, Berry phase."""
    # number of occupied states
    nocc=wf.shape[1]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0]-1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j,k]=_wf_dpr(wf[i,j,:],wf[i+1,k,:])
        # only find Berry phase
        if berry_evals==False:
            # multiply overlap matrices
            prd=np.dot(prd,ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU,sing,matV=np.linalg.svd(ovr)
            prd=np.dot(prd,np.dot(matU,matV))
    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(prd)
        pha=(-1.0)*np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(prd)
        eval_pha=(-1.0)*np.angle(evals)
        # sort these numbers as well
        eval_pha=np.sort(eval_pha)
        return eval_pha

def _one_flux_plane(wfs2d):
    "Compute fluxes on a two-dimensional plane of states."
    # size of the mesh
    nk0=wfs2d.shape[0]
    nk1=wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd=wfs2d.shape[2]

    # here store flux through each plaquette of the mesh
    all_phases=np.zeros((nk0-1,nk1-1),dtype=float)

    # go over all plaquettes
    for i in range(nk0-1):
        for j in range(nk1-1):
            # generate a small loop made out of four pieces
            wf_use=[]
            wf_use.append(wfs2d[i,j])
            wf_use.append(wfs2d[i+1,j])
            wf_use.append(wfs2d[i+1,j+1])
            wf_use.append(wfs2d[i,j+1])
            wf_use.append(wfs2d[i,j])
            wf_use=np.array(wf_use,dtype=complex)
            # calculate phase around one plaquette
            all_phases[i,j]=_one_berry_loop(wf_use)

    return all_phases

def no_2pi(x,clos):
    "Make x as close to clos by adding or removing 2pi"
    while abs(clos-x)>np.pi:
        if clos-x>np.pi:
            x+=2.0*np.pi
        elif clos-x<-1.0*np.pi:
            x-=2.0*np.pi
    return x

def _one_phase_cont(pha,clos):
    """Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible."""
    ret=np.copy(pha)
    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1]
        # make sure there are no 2pi jumps
        ret[i]=no_2pi(ret[i],cmpr)
    return ret

def _array_phases_cont(arr_pha,clos):
    """Reads in 2d array of phases *arr_pha* and makes sure that they
    are continuous along first index, i.e., that there are no jumps of
    2pi. First array of phasese is made as close to *clos* as
    possible."""
    ret=np.zeros_like(arr_pha)
    # go over all points
    for i in range(arr_pha.shape[0]):
        # which phases to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1,:]
        # remember which indices are still available to be matched
        avail=list(range(arr_pha.shape[1]))
        # go over all phases in cmpr[:]
        for j in range(cmpr.shape[0]):
            # minimal distance between pairs
            min_dist=1.0E10
            # closest index
            best_k=None
            # go over each phase in arr_pha[i,:]
            for k in avail:
                cur_dist=np.abs(np.exp(1.0j*cmpr[j])-np.exp(1.0j*arr_pha[i,k]))
                if cur_dist<=min_dist:
                    min_dist=cur_dist
                    best_k=k
            # remove this index from being possible pair later
            avail.pop(avail.index(best_k))
            # store phase in correct place
            ret[i,j]=arr_pha[i,best_k]
            # make sure there are no 2pi jumps
            ret[i,j]=no_2pi(ret[i,j],cmpr[j])
    return ret


def _cart_to_red(tmp,cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    (a1,a2,a3)=tmp
    # matrix with lattice vectors
    cnv=np.array([a1,a2,a3])
    # transpose a matrix
    cnv=cnv.T
    # invert a matrix
    cnv=np.linalg.inv(cnv)
    # reduced coordinates
    red=np.zeros_like(cart,dtype=float)
    for i in range(0,len(cart)):
        red[i]=np.dot(cnv,cart[i])
    return red

def _red_to_cart(tmp,red):
    "Convert reduced to cartesian vectors."
    (a1,a2,a3)=tmp
    # cartesian coordinates
    cart=np.zeros_like(red,dtype=float)
    for i in range(0,len(cart)):
        cart[i,:]=a1*red[i][0]+a2*red[i][1]+a3*red[i][2]
    return cart

def _is_int(a):
    return np.issubdtype(type(a), np.integer)

def _offdiag_approximation_warning_and_stop():
    raise Exception("""

----------------------------------------------------------------------

  It looks like you are trying to calculate Berry-like object that
  involves position operator.  However, you are using a tight-binding
  model that was generated from Wannier90.  This procedure introduces
  approximation as it ignores off-diagonal elements of the position
  operator in the Wannier basis.  This is discussed here in more
  detail:

    http://www.physics.rutgers.edu/pythtb/usage.html#pythtb.w90

  If you know what you are doing and wish to continue with the
  calculation despite this approximation, please call the following
  function on your tb_model object

    my_model.ignore_position_operator_offdiagonal()

----------------------------------------------------------------------

""")
    
