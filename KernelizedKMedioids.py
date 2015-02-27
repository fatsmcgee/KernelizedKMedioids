import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_digits

def get_gram_matrix(data,kernel_func):
    n = data.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        print(i)
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
    

def get_point_set(image):
    rows,cols = image.shape
    X,Y = np.meshgrid(range(cols),range(rows))
    
    image_present = (image>0).flatten()
    x_coords = X.flatten()[image_present]
    y_coords = Y.flatten()[image_present]
    return np.vstack((x_coords,y_coords)).T
    

    
    
def fit_mv_gaussian(point_set):
    mu = np.atleast_2d(np.mean(point_set,0))
    n = point_set.shape[0]
    
    mean_subtracted = point_set - mu
    sigma = (1/n)*np.dot(mean_subtracted.T,mean_subtracted)
    return mu,sigma
    
    
def make_kl_divergence_kernel(im_shape):
    def kl_divergence_kernel(im1,im2):
        #convert each image to a bag of coordinates 
        im1 = im1.reshape(im_shape)
        im2 = im2.reshape(im_shape)
        
        s1 = get_point_set(im1)
        D = s1.shape[1]
        mu1,sigma1 = fit_mv_gaussian(s1)
        
        s2 = get_point_set(im2)
        mu2,sigma2 = fit_mv_gaussian(s2)
        
        #first calculate symmetric divergence of two distributions
        sym_divergence = np.trace(sigma1 * inv(sigma2))
        sym_divergence += np.trace(sigma2 * inv(sigma1))
        sym_divergence -= 2*D
        sym_divergence += np.trace(np.matrix(inv(sigma1) + inv(sigma2))\
                    *np.matrix((mu1.T-mu2.T))*np.matrix((mu1-mu2)))
        
        #kernel is exp of negative symmetric divergence
        return np.exp(-sym_divergence)
        
    return kl_divergence_kernel


def mnist_medioids():
    mnist = load_digits()
    n_images = mnist.images.shape[0]
    
    #use every 5 digits
    indices = range(0,n_images,5)
    data = mnist.data
    data = data[indices,:]
    
    #targets = mnist.target[indices,:]
    k = 10
    #kern = make_kl_divergence_kernel((8,8))
    kern = linear_kernel
    medioids,assignments = get_k_medioids(data,10,kern)
    
    for i in range(k):
        plt.subplot(5,2,i)
        im = data[medioids[i],:].reshape((8,8))
        plt.imshow(im, cmap = cm.Greys_r)
    #return medioids


mnist_medioids()


   