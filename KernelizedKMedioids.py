import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_digits

def get_gram_matrix(data,kernel_func):
    n = data.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel_func(data[i,:],data[j,:])
            
    print("gram done")
    return K
        
def get_distances_matrix(data,kernel_func):
    K = get_gram_matrix(data,kernel_func)
    n = data.shape[0]
    D = np.zeros((n,n))    
    for i in range(n):
        for j in range(n):
            D[i,j] = K[i,i] - 2*K[i,j] + K[j,j]
    print ("distances done")            
    return D
    
def get_k_medioids(data,k,kernel_func,iterations=50):
    D = get_distances_matrix(data,kernel_func)
    n = data.shape[0]
    
    #intialize medioids randomly, get first k
    medioids = range(n)
    medioids = np.random.permutation(medioids)
    medioids = medioids[:k]
    
    assignments = np.zeros(n)
    
    for iter_idx in range(iterations):
        #update assignments
        for i in range(n):
            #find the medioid that is closest to each data point
            assignments[i] = np.argmin([D[i,medioids[j]] for j in range(len(medioids))])
			
        #update medioids
        for clust in range(k):
            #data indices of cluster clust
            cluster = [idx for idx,c in enumerate(assignments) if c==clust]
            cluster_size = len(cluster)
            
			#for each data point in cluster, total dist to all other data points 
            #in cluster
            total_dists = np.zeros(cluster_size)
            for j in range(len(cluster)):
                total_dists[j] = sum(D[cluster[j],cluster[j2]] \
                                    for j2 in range(cluster_size) if j2!=j)
            #print("Best dist for cluster {0} = {1}".format(clust,min(total_dists)))
            best_medioid = np.argmin(total_dists)
            medioids[clust] = cluster[best_medioid]
    return medioids,assignments
    
#todo: try fancy kernels like pyramid matching, KL-divergence kernel
def linear_kernel(dig1,dig2):
    return np.dot(dig1,dig2)
    

def mnist_medioids():
    mnist = load_digits()
    n_images = mnist.images.shape[0]
    
    #use every 5 digits
    indices = range(0,n_images,5)
    data = mnist.data
    data = data[indices,:]
    
    targets = mnist.target[indices,:]
    k = 10
    medioids,assignments = get_k_medioids(data,10,linear_kernel)
    
    for i in range(k):
        plt.subplot(5,2,i)
        im = data[medioids[i],:].reshape((8,8))
        plt.imshow(im, cmap = cm.Greys_r)
    #return medioids





   