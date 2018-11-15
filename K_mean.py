import time
import math
import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class K_means:
    """ K-means cluster
    Args:
        k (int): number of clusters
        tol (float): minimum tolerance in optimizing centroids
        max_iter (int): maximum number of steps to optimizing the centroids

    Attributes:
        fit (void): Train this k-mean cluster and find the k-centroids
        labels (1D-array): List of labels that assigned to each point in the dataset
    """
    def __init__(self, k=2, tol=1e-3, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, data):
        self.centroids = []
        
        # Initialize with two random centroids
        random_index = random.sample(range(0, len(data)), self.k)
        for i in range(self.k):
            self.centroids.append(data[random_index[i]])
        
        
        # Optimize in each iteration and update the centroids until hit the tolerance or max_iter
        for i in range(self.max_iter):
            self.labels = {}
            
            for i in range(self.k):
                self.labels[i] = []
                
            for point in data:
                # Calculate the distance from the point to each centroid
                distance_list = [math.hypot(point[0] - self.centroids[i][0], point[1] - self.centroids[i][1]) for i in range(self.k)]
                # Get the smallest distance index of centroid as label
                label = distance_list.index(min(distance_list))
                self.labels[label].append(point)
            
            prev_centroids = self.centroids[:]
            
            for label in self.labels:
                # Calculate and update the new centroid
                self.centroids[label] = np.average(self.labels[label], axis=0)
            
            # Calculate the difference from the last centroid, break the loop if within the tolerance
            if (np.allclose(prev_centroids, self.centroids, atol=self.tol)):
                break
                    
    def labels_(self, data):
        labels = []
        for point in data:
            distance_list = [math.hypot(point[0] - self.centroids[i][0], point[1] - self.centroids[i][1]) for i in range(self.k)]
            label = distance_list.index(min(distance_list))
            labels.append(label)
        return labels

def load_data():
    """ Load the .mat files from the Dataset

    Returns: 
        data: a collection of 2-d points
    """
    data = sio.loadmat("./data/dataset")["Points"]
    
    return data

def vis_cluster(data, labels, ax):
    """
    Show the clustering results given original points and labels
    """
    # get the distinct values of labels
    label_list = list(set(labels))

    # normalize the labels in order to map with the colormap
    norm = Normalize(vmin=0, vmax=len(label_list))
    
    # Plot points with different colors for different clusters
    for index in range(len(data)):
        ax.scatter(x=data[index][0],y=data[index][1], color=cm.jet(norm(labels[index])))

def write_output(path, data, labels):
    """
    Wite the data and clustering results to the txt file
    """
    with open(path, "w") as text_file:
        text_file.write("point id, x-coordinate, y-coordinate, cluster id \n")
        for index in range(len(data)):
            text_file.write('{}, {}, {}, {} \n'.format(index+1, data[index][0], data[index][1], labels[index]))

def main():
    # Load the data
    data = load_data()

    k_candidates = [2, 10, 20, 30]
    labels_all = []
    for k in k_candidates:
        startTime = time.time()

        # Biuld the DBSCAN cluster and fit the data
        clustering = K_means(k=k)
        clustering.fit(data)
        
        # Get the labels for each point
        labels = clustering.labels_(data)
        labels_all.append(labels)

        # Write to txt file
        path = "output_KMean_k=%i.txt"%k
        write_output(path, data, labels)

        print("When k = %i, the running time is: %f seconds" % (k, time.time()-startTime))

    
    # visualize the clustered points with matplot
    f, axarr = plt.subplots(2, 2, figsize=(12,12))
    f.suptitle('K-means Clustering')
    for i in range(len(k_candidates)):
        vis_cluster(data, labels_all[i], axarr[int(i/2), int(i%2)])
        axarr[int(i/2), int(i%2)].set_title('K = %i' % k_candidates[i])
    plt.show()
    

if __name__ == "__main__":
    main()