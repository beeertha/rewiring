import numpy as np
from sklearn.utils.validation import check_random_state
import scipy.spatial.distance as cdist

# create a fake connectivity and distance matrix
def fake_matrices(N, ndisc):
    
    # undirected fully connected network
    W = np.zeros([N,N])
    xs,ys = np.triu_indices(N,k=1)
    vals = np.random.random(len(xs))
    W[xs,ys] = vals
    W[ys,xs] = vals
    # disconnect some edges
    for i in range(ndisc):
        r1 = np.random.randint(N)
        r2 = np.random.randint(N)
        W[r1,r2] = 0
        W[r2,r1] = 0

    # distance between nodes
    pos = 5* np.random.random((N,3))
    D = cdist.cdist(pos, pos)
    
    return W, D

# bin the edges according to their lengths
def dist_label(D, nbins):
    
    N = len(D)
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N,N))
    
    for n in range(nbins):
        i,j = np.where(np.logical_and(bins[n] <= D, D < bins[n+1]))
        L[i,j] = n+1
    
    return L

# rewire an undirected weigthed network preserving degree and lengths
def rew_kd(W, D, nbins, nrew, replacement=False):
    
    """
    Parameters
    ----------
	W: connectivity matrix
	D: distance matrix 
	nbins: number of bins in which the connections length is divided
	nrew: number of edge swaps

    Returns
    -------

	newW: weighted rewired matrix

    References
    ----------
    Richard Betzel rewiring matching lenght and degree & Ross Markello python translation
    """  
        
    seed = 4
    rs = check_random_state(seed)
    
    N = len(W)
    # bin the distance
    L = dist_label(D, nbins)
    # binarized connectivity
    B = np.zeros((N,N))
    B[W!=0] = 1
    # existing edges
    cn_x, cn_y = np.where(np.triu((B!=0)*B, k=1))
    
    tries = 0
    nr = 0
    newB = np.copy(B)    
    
    while((len(cn_x) >= 2) & (nr < nrew)):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r],cn_y[r]
        tries += 1
        
        # options to rewire with: the connected ones that doesn't involve (n_x,n_y)
        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if(len(np.where(index)[0]) == 0):
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)
            
        else: 
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            # options that will preserve the distances
            # (ops1_x, ops1_y) such that L(n_x,n_y) = L(n_x, ops1_x) & L(ops1_x,ops1_y) = L(n_y, ops1_y)
            index = (L[n_x,n_y] == L[n_x,ops1_x]) & (L[ops1_x, ops1_y] == L[n_y,ops1_y])
            
            if(len(np.where(index)[0]) == 0):
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)
                
            else: 
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [(newB[min(n_x,ops2_x[i])][max(n_x,ops2_x[i])] == 0) & 
                         (newB[min(n_y,ops2_y[i])][max(n_y,ops2_y[i])] == 0) for i in range(len(ops2_x))]
                
                if(len(np.where(index)[0]) == 0):
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)
    
                else: 
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]
                    
                    # choose randomly one from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]
    
                    # Disconnect the existing edges
                    newB[n_x,n_y] = 0
                    newB[nn_x,nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x,nn_x),max(n_x,nn_x)] = 1
                    newB[min(n_y,nn_y),max(n_y,nn_y)] = 1
                    # one successfull rewire!
                    nr += 1
                    
                    # used edges from cn_x, cn_y
                    index = np.where((cn_x == n_x) & (cn_y == n_y))[0]
                    # rewire with replacement
                    if replacement:
                        cn_y[index] =  nn_x
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index] = n_y
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)
                        
    if(nr < nrew):
        print(f"Out of rewirable edges: {len(cn_x)}")
        
    print(f"number of successful rewires: {nr}, from: {tries} number of tries")
    
    i,j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j,i] = newB[i,j]
    
    # check the number of edges is preserved
    if(len(np.where(B!=0)[0]) != len(np.where(newB!=0)[0])):
        print(f"ERROR --- B: {len(np.where(B!=0)[0])}, newB:{len(np.where(newB!=0)[0])}")
    # check that the degree of the nodes its the same
    for i in range(N):
        if(np.sum(B[i]) != np.sum(newB[i])):
            print(f"ERROR --- node {i} change k: {np.sum(B[i]) - np.sum(newB[i])}")
    
    # Reassign the weights
    mask = np.triu(B!=0, k=1)
    inids = D[mask]
    iniws = W[mask]
    inids_index = np.argsort(inids)
    # Weights corresponding to the shortest to largest edge
    iniws = iniws[inids_index]
    mask = np.triu(newB != 0, k=1)
    finds = D[mask]
    i,j = np.where(mask)
    # Sort the new edges from the shortest to the largest
    finds_index = np.argsort(finds)
    i_sort = i[finds_index]
    j_sort = j[finds_index]
    newW = np.zeros((N,N))
    # Assign the initial sorted weights
    newW[i_sort,j_sort] = iniws
    # Make it symmetrical
    newW[j_sort,i_sort] = iniws

    return newW



# example:
N = 6
ndisc = 5
nbins = 1
nrew = 2

W, D = fake_matrices(N, ndisc)
newW = rew_kd(W, D, nbins, nrew, replacement=False)


