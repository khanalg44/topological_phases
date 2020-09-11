from __future__ import print_function

__version__='1.7.2'

import numpy as np # numerics for matrices
import sys # for exiting
import copy # for deepcopying

class tb_model(object):
    def __init__(self,dim_k,dim_r,lat=None,orb=None,per=None,nspin=1):
        self._dim_k=dim_k
        self._dim_r=dim_r
        self._lat=np.array(lat,dtype=float)
        self._norb=self._orb.shape[0] # number of orbitals
        self._per=per
        self._nspin=nspin
        self._assume_position_operator_diagonal=True
        self._nsta=self._norb*self._nspin
        self._site_energies=np.zeros((self._norb),dtype=float)
        self._site_energies=np.zeros((self._norb,2,2),dtype=complex)
        self._site_energies_specified=np.zeros(self._norb,dtype=bool)
        self._site_energies_specified[:]=False
        self._hoppings=[]

    def set_onsite(self,onsite_en,ind_i=None,mode="set"):
        # specifying onsite energies from scratch, can be called only once
        self._site_energies[ind_i]=self._val_to_block(onsite_en)
        self._site_energies_specified[ind_i]=True
        self._site_energies[i]=self._val_to_block(onsite_en[i])
        self._site_energies_specified[:]=True
        
    def set_hop(self,hop_amp,ind_i,ind_j,ind_R=None,mode="set",allow_conjugate_pair=False):
        #
        tmpR=np.zeros(self._dim_r,dtype=int)
        tmpR[self._per]=ind_R
        ind_R=tmpR
        hop_use=self._val_to_block(hop_amp)
        new_hop=[hop_use,int(ind_i),int(ind_j),np.array(ind_R)]
        self._hoppings[use_index]=new_hop
        elif mode.lower()=="add":
                self._hoppings[use_index][0]+=new_hop[0]

    def get_num_orbitals(self):
        return self._norb
    def get_orb(self):
        return self._orb.copy()
    def get_lat(self):
        return self._lat.copy()

    def _gen_ham(self,k_input=None):
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
                ham=self._gen_ham(k)
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
        if type(num).__name__!='int':
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

    def make_supercell(self, sc_red_lat, return_sc_vectors=False, to_home=True):
        # Can't make super cell for model without periodic directions
        if self._dim_r==0:
            raise Exception("\n\nMust have at least one periodic direction to make a super-cell")
        
        # convert array to numpy array
        use_sc_red_lat=np.array(sc_red_lat)
        
        # checks on super-lattice array
        if use_sc_red_lat.shape!=(self._dim_r,self._dim_r):
            raise Exception("\n\nDimension of sc_red_lat array must be dim_r*dim_r")
        if use_sc_red_lat.dtype!=int:
            raise Exception("\n\nsc_red_lat array elements must be integers")
        for i in range(self._dim_r):
            for j in range(self._dim_r):
                if (i==j) and (i not in self._per) and use_sc_red_lat[i,j]!=1:
                    raise Exception("\n\nDiagonal elements of sc_red_lat for non-periodic directions must equal 1.")
                if (i!=j) and ((i not in self._per) or (j not in self._per)) and use_sc_red_lat[i,j]!=0:
                    raise Exception("\n\nOff-diagonal elements of sc_red_lat for non-periodic directions must equal 0.")
        if np.abs(np.linalg.det(use_sc_red_lat))<1.0E-6:
            raise Exception("\n\nSuper-cell lattice vectors length/area/volume too close to zero, or zero.")
        if np.linalg.det(use_sc_red_lat)<0.0:
            raise Exception("\n\nSuper-cell lattice vectors need to form right handed system.")

        # converts reduced vector in original lattice to reduced vector in super-cell lattice
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(np.array(use_sc_red_lat.T,dtype=float),
                                   np.array(red_vec_orig,dtype=float))

        # conservative estimate on range of search for super-cell vectors
        max_R=np.max(np.abs(use_sc_red_lat))*self._dim_r

        # candidates for super-cell vectors
        # this is hard-coded and can be improved!
        sc_cands=[]
        if self._dim_r==1:
            for i in range(-max_R,max_R+1):
                sc_cands.append(np.array([i]))
        elif self._dim_r==2:
            for i in range(-max_R,max_R+1):
                for j in range(-max_R,max_R+1):
                    sc_cands.append(np.array([i,j]))
        elif self._dim_r==3:
            for i in range(-max_R,max_R+1):
                for j in range(-max_R,max_R+1):
                    for k in range(-max_R,max_R+1):
                        sc_cands.append(np.array([i,j,k]))
        elif self._dim_r==4:
            for i in range(-max_R,max_R+1):
                for j in range(-max_R,max_R+1):
                    for k in range(-max_R,max_R+1):
                        for l in range(-max_R,max_R+1):
                            sc_cands.append(np.array([i,j,k,l]))
        else:
            raise Exception("\n\nWrong dimensionality of dim_r!")

        # find all vectors inside super-cell
        # store them here
        sc_vec=[]
        eps_shift=np.sqrt(2.0)*1.0E-8 # shift of the grid, so to avoid double counting
        #
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red=to_red_sc(vec).tolist()
            # check if in the interior
            inside=True
            for t in tmp_red:
                if t<=-1.0*eps_shift or t>1.0-eps_shift:
                    inside=False                
            if inside==True:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc=len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(use_sc_red_lat))))!=num_sc:
            raise Exception("\n\nSuper-cell generation failed! Wrong number of super-cell vectors found.")

        # cartesian vectors of the super lattice
        sc_cart_lat=np.dot(use_sc_red_lat,self._lat)

        # orbitals of the super-cell tight-binding model
        sc_orb=[]
        for cur_sc_vec in sc_vec: # go over all super-cell vectors
            for orb in self._orb: # go over all orbitals
                # shift orbital and compute coordinates in
                # reduced coordinates of super-cell
                sc_orb.append(to_red_sc(orb+cur_sc_vec))

        # create super-cell tb_model object to be returned
        sc_tb=tb_model(self._dim_k,self._dim_r,sc_cart_lat,sc_orb,per=self._per,nspin=self._nspin)

        # remember if came from w90
        sc_tb._assume_position_operator_diagonal=self._assume_position_operator_diagonal

        # repeat onsite energies
        for i in range(num_sc):
            for j in range(self._norb):
                sc_tb.set_onsite(self._site_energies[j],i*self._norb+j)

        # set hopping terms
        for c,cur_sc_vec in enumerate(sc_vec): # go over all super-cell vectors
            for h in range(len(self._hoppings)): # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp=self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R=copy.deepcopy(self._hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part=np.floor(to_red_sc(ind_R+cur_sc_vec)) # round down!
                sc_part=np.array(sc_part,dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part=ind_R+cur_sc_vec-np.dot(sc_part,use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind=None
                for p,pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec==orig_part):
                        if pair_ind!=None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind=p
                if pair_ind==None:
                    raise Exception("\n\nDid not find super cell vector!")
                        
                # index of "from" and "to" hopping indices
                hi=self._hoppings[h][1] + c*self._norb
                hj=self._hoppings[h][2] + pair_ind*self._norb
                
                # add hopping term
                sc_tb.set_hop(amp,hi,hj,sc_part,mode="add",allow_conjugate_pair=True)

        # put orbitals to home cell if asked for
        if to_home==True:
            sc_tb._shift_to_home()

        # return new tb model and vectors if needed
        if return_sc_vectors==False:
            return sc_tb
        else:
            return (sc_tb,sc_vec)

    def _shift_to_home(self):
        
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
        if type(to_remove).__name__=='int':
            orb_index=[to_remove]
        else:
            orb_index=copy.deepcopy(to_remove)

        # check range of indices
        for i,orb_ind in enumerate(orb_index):
            if orb_ind < 0 or orb_ind > self._norb-1 or type(orb_ind).__name__!='int':
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
        # return new model
        return ret


    def k_uniform_mesh(self,mesh_size):
        
        # get the mesh size and checks for consistency
        use_mesh=np.array(list(map(round,mesh_size)),dtype=int)
        if use_mesh.shape!=(self._dim_k,):
            print(use_mesh.shape)
            raise Exception("\n\nIncorrect size of the specified k-mesh!")
        if np.min(use_mesh)<=0:
            raise Exception("\n\nMesh must have positive non-zero number of elements.")

        # construct the mesh
        if self._dim_k==1:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[1])
            norm=norm.transpose([1,0])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,0]).reshape([use_mesh[0],1])
        elif self._dim_k==2:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[2])
            norm=norm.transpose([2,0,1])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,2,0]).reshape([use_mesh[0]*use_mesh[1],2])
        elif self._dim_k==3:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[3])
            norm=norm.transpose([3,0,1,2])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,2,3,0]).reshape([use_mesh[0]*use_mesh[1]*use_mesh[2],3])
        else:
            raise Exception("\n\nUnsupported dim_k!")

        return k_vec

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
    
        if report==True:
            print()

        return (k_vec,k_dist,k_node)

    def ignore_position_operator_offdiagonal(self):
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
        if self._norb<=1:
            all_gaps=None # trivial case since there is only one band
        else:
            gap_dim=np.copy(self._mesh_arr)-1
            gap_dim=np.append(gap_dim,self._norb*self._nspin-1)
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

        return all_gaps.min(axis=tuple(range(self._dim_arr)))

    def __check_key(self,key):
        # do some checks for 1D
        if self._dim_arr==1:
            if type(key).__name__!='int':
                raise TypeError("Key should be an integer!")
            if key<(-1)*self._mesh_arr[0] or key>=self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do checks for higher dimension
        else:
            if len(key)!=self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i,k in enumerate(key):
                if type(k).__name__!='int':
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
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_matrix(evec,dir)

    def position_expectation(self, key, occ, dir):
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_expectation(evec,dir)

    def position_hwf(self, key, occ, dir, hwf_evec=False, basis="bloch"):
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
        print(""" 

Warning:
  Usage of function berry_curv is discouraged.
  It has been renamed as berry_flux, which should be used instead.
""")
        return self.berry_flux(occ,individual_phases)



def k_path(kpts,nk,endpoint=True):
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


class w90(object):

    def __init__(self,path,prefix):
        # store path and prefix
        self.path=path
        self.prefix=prefix

        # read in lattice_vectors
        f=open(self.path+"/"+self.prefix+".win","r")
        ln=f.readlines()
        f.close()
        # get lattice vector
        self.lat=np.zeros((3,3),dtype=float)
        found=False
        for i in range(len(ln)):
            sp=ln[i].split()
            if len(sp)>=2:
                if sp[0].lower()=="begin" and sp[1].lower()=="unit_cell_cart":
                    # get units right
                    if ln[i+1].strip().lower()=="bohr":
                        pref=0.5291772108
                        skip=1
                    elif ln[i+1].strip().lower() in ["ang","angstrom"]:
                        pref=1.0
                        skip=1
                    else:
                        pref=1.0
                        skip=0
                    # now get vectors
                    for j in range(3):
                        sp=ln[i+skip+1+j].split()
                        for k in range(3):
                            self.lat[j,k]=float(sp[k])*pref
                    found=True
                    break
        if found==False:
            raise Exception("Unable to find unit_cell_cart block in the .win file.")

        # read in hamiltonian matrix, in eV
        f=open(self.path+"/"+self.prefix+"_hr.dat","r")
        ln=f.readlines()
        f.close()
        #
        # get number of wannier functions
        self.num_wan=int(ln[1])
        # get number of Wigner-Seitz points
        num_ws=int(ln[2])
        # get degenereacies of Wigner-Seitz points
        deg_ws=[]
        for j in range(3,len(ln)):
            sp=ln[j].split()
            for s in sp:
                deg_ws.append(int(s))
            if len(deg_ws)==num_ws:
                last_j=j
                break
            if len(deg_ws)>num_ws:
                raise Exception("Too many degeneracies for WS points!")
        deg_ws=np.array(deg_ws,dtype=int)
        # now read in matrix elements
        # Convention used in w90 is to write out:
        # R1, R2, R3, i, j, ham_r(i,j,R)
        # where ham_r(i,j,R) corresponds to matrix element < i | H | j+R >
        self.ham_r={} # format is ham_r[(R1,R2,R3)]["h"][i,j] for < i | H | j+R >
        ind_R=0 # which R vector in line is this?
        for j in range(last_j+1,len(ln)):
            sp=ln[j].split()
            # get reduced lattice vector components
            ham_R1=int(sp[0])
            ham_R2=int(sp[1])
            ham_R3=int(sp[2])
            # get Wannier indices
            ham_i=int(sp[3])-1
            ham_j=int(sp[4])-1
            # get matrix element
            ham_val=float(sp[5])+1.0j*float(sp[6])
            # store stuff, for each R store hamiltonian and degeneracy
            ham_key=(ham_R1,ham_R2,ham_R3)
            if (ham_key in self.ham_r)==False:
                self.ham_r[ham_key]={
                    "h":np.zeros((self.num_wan,self.num_wan),dtype=complex),
                    "deg":deg_ws[ind_R]
                    }
                ind_R+=1
            self.ham_r[ham_key]["h"][ham_i,ham_j]=ham_val

        # check if for every non-zero R there is also -R
        for R in self.ham_r:
            if not (R[0]==0 and R[1]==0 and R[2]==0):
                found_pair=False
                for P in self.ham_r:
                    if not (R[0]==0 and R[1]==0 and R[2]==0):
                        # check if they are opposite
                        if R[0]==-P[0] and R[1]==-P[1] and R[2]==-P[2]: 
                            if found_pair==True:
                                raise Exception("Found duplicate negative R!")
                            found_pair=True
                if found_pair==False:
                    raise Exception("Did not find negative R for R = "+R+"!")

        # read in wannier centers
        f=open(self.path+"/"+self.prefix+"_centres.xyz","r")
        ln=f.readlines()
        f.close()
        # Wannier centers in Cartesian, Angstroms
        xyz_cen=[]
        for i in range(2,2+self.num_wan):
            sp=ln[i].split()
            if sp[0]=="X":
                tmp=[]
                for j in range(3):
                    tmp.append(float(sp[j+1]))
                xyz_cen.append(tmp)
            else:
                raise Exception("Inconsistency in the centres file.")
        self.xyz_cen=np.array(xyz_cen,dtype=float)
        # get orbital positions in reduced coordinates
        self.red_cen=_cart_to_red((self.lat[0],self.lat[1],self.lat[2]),self.xyz_cen)


    def model(self,zero_energy=0.0,min_hopping_norm=None,max_distance=None,ignorable_imaginary_part=None):
        # make the model object
        tb=tb_model(3,3,self.lat,self.red_cen)

        # remember that this model was computed from w90
        tb._assume_position_operator_diagonal=False

        # add onsite energies
        onsite=np.zeros(self.num_wan,dtype=float)
        for i in range(self.num_wan):
            tmp_ham=self.ham_r[(0,0,0)]["h"][i,i]/float(self.ham_r[(0,0,0)]["deg"])
            onsite[i]=tmp_ham.real
            if np.abs(tmp_ham.imag)>1.0E-9:
                raise Exception("Onsite terms should be real!")
        tb.set_onsite(onsite-zero_energy)

        # add hopping terms
        for R in self.ham_r:
            # avoid double counting
            use_this_R=True
            # avoid onsite terms
            if R[0]==0 and R[1]==0 and R[2]==0:
                avoid_diagonal=True
            else:
                avoid_diagonal=False
                # avoid taking both R and -R
                if R[0]!=0:
                    if R[0]<0:
                        use_this_R=False
                else:
                    if R[1]!=0:
                        if R[1]<0:
                            use_this_R=False
                    else:
                        if R[2]<0:
                            use_this_R=False
            # get R vector
            vecR=_red_to_cart((self.lat[0],self.lat[1],self.lat[2]),[R])[0]
            # scan through unique R
            if use_this_R==True:
                for i in range(self.num_wan):
                    vec_i=self.xyz_cen[i]
                    for j in range(self.num_wan):
                        vec_j=self.xyz_cen[j]
                        # get distance between orbitals
                        dist_ijR=np.sqrt(np.dot(-vec_i+vec_j+vecR,
                                                -vec_i+vec_j+vecR))
                        # to prevent double counting
                        if not (avoid_diagonal==True and j<=i):
                            
                            # only if distance between orbitals is small enough
                            if max_distance is not None:
                                if dist_ijR>max_distance:
                                    continue

                            # divide the matrix element from w90 with the degeneracy
                            tmp_ham=self.ham_r[R]["h"][i,j]/float(self.ham_r[R]["deg"])

                            # only if big enough matrix element
                            if min_hopping_norm is not None:
                                if np.abs(tmp_ham)<min_hopping_norm:
                                    continue

                            # remove imaginary part if needed
                            if ignorable_imaginary_part is not None:
                                if np.abs(tmp_ham.imag)<ignorable_imaginary_part:
                                    tmp_ham=tmp_ham.real+0.0j

                            # set the hopping term
                            tb.set_hop(tmp_ham,i,j,list(R))

        return tb



    def dist_hop(self):

        ret_ham=[]
        ret_dist=[]
        for R in self.ham_r:
            # treat diagonal terms differently
            if R[0]==0 and R[1]==0 and R[2]==0:
                avoid_diagonal=True
            else:
                avoid_diagonal=False

            # get R vector
            vecR=_red_to_cart((self.lat[0],self.lat[1],self.lat[2]),[R])[0]
            for i in range(self.num_wan):
                vec_i=self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j=self.xyz_cen[j]
                    # diagonal terms
                    if not (avoid_diagonal==True and i==j):
                        
                        # divide the matrix element from w90 with the degeneracy
                        ret_ham.append(self.ham_r[R]["h"][i,j]/float(self.ham_r[R]["deg"]))

                        # get distance between orbitals
                        ret_dist.append(np.sqrt(np.dot(-vec_i+vec_j+vecR,-vec_i+vec_j+vecR)))

        return (np.array(ret_dist),np.array(ret_ham))


    def shells(self,num_digits=2):
        shells=[]
        for R in self.ham_r:
            # get R vector
            vecR=_red_to_cart((self.lat[0],self.lat[1],self.lat[2]),[R])[0]
            for i in range(self.num_wan):
                vec_i=self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j=self.xyz_cen[j]
                    # get distance between orbitals
                    dist_ijR=np.sqrt(np.dot(-vec_i+vec_j+vecR,
                                             -vec_i+vec_j+vecR))
                    # round it up
                    shells.append(round(dist_ijR,num_digits))

        # remove duplicates and sort
        shells=np.sort(list(set(shells)))

        return shells


    def w90_bands_consistency(self):

        # read in kpoints in reduced coordinates
        kpts=np.loadtxt(self.path+"/"+self.prefix+"_band.kpt",skiprows=1)
        # ignore weights
        kpts=kpts[:,:3]

        # read in energies
        ene=np.loadtxt(self.path+"/"+self.prefix+"_band.dat")
        # ignore kpath distance
        ene=ene[:,1]
        # correct shape 
        ene=ene.reshape((self.num_wan,kpts.shape[0]))

        return (kpts,ene)


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

def _offdiag_approximation_warning_and_stop():
    raise Exception("""
""")
    
