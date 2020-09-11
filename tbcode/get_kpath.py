import numpy as np

def get_k_path(kpts, nk, lat):

    #kvec, kdist, knode=k_path(path, nk, lat)
    
    dim_k= len(lat[0])
    lat_per = np.array(lat);
    k_list=np.array(kpts)
    n_nodes=k_list.shape[0]
    k_metric = np.linalg.inv(np.dot(lat_per,lat_per.T))
    
    k_node=np.zeros(n_nodes,dtype=float)
    for n in range(1,n_nodes):
        dk = k_list[n]-k_list[n-1]
        dklen = np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
        k_node[n]=k_node[n-1]+dklen

    node_index=[0]
    for n in range(1,n_nodes-1):
        frac=k_node[n]/k_node[-1]
        node_index.append(int(round(frac*(nk-1))))
    node_index.append(nk-1)

    k_dist=np.zeros(nk,dtype=float)
    k_vec=np.zeros((nk,dim_k),dtype=float)

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

def get_kpts_3D(path, nk=11, Print=False):
    lat = [ [1., 0., 0.], [0., 1., 0.], [0., 0., 1.5] ]
    kvec, kdist, knode = get_k_path(path, nk, lat)
    if Print:
        print (kvec)
        print (kdist ,knode)

    return (kvec, kdist, knode)


def get_kpts_2D(path, nk, Print=False):
    lat = [ [1., 0.], [0., 1.]]
    kvec, kdist, knode =get_k_path(path, nk, lat)
    if Print:        
        print (kvec)
        print (kdist ,knode)

    return (kvec, kdist, knode)

if __name__=="__main__":
    nk=11
    path = [ [0., 0.], [0.5, 0.0], [0.5, 0.5], [0., 0.] ]
    get_kpts_2D(path, nk, Print=True)
    path = [ [0., 0., 0.], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5] ]
    get_kpts_3D(path, nk, Print=True)
