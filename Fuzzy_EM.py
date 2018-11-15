import scipy.io as sio
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class fuzzy_cluster:
    """ Fuzzy cluster
    Args:
        tol (float): minimum tolerance in optimizing centroids
        max_iter (int): maximum number of steps to optimizing the centroids

    Attributes:
        distance (float): compute the distance of two points
        compute_wt (float): compute the wait for one point assign to different clusters
        update_centroid (1D-array): Update the centroids given the data and fuzzy labels
        compute_sse (float): compute the sum of squared error (SSE)
        fit (void): Train this fuzzy cluster and find the centroids
        labels (2D-array, 1D-array): List of fuzzy labels that assigned to each point in the dataset, and get the fianl centroid
    """
    def __init__(self, tol=1e-3, max_iter=200):
        self.k = 2
        self.tol = tol
        self.max_iter = max_iter
    
    def distance(self, x, y):
        return math.hypot(x[0] - y[0], x[1] - y[1])
    
    def compute_wt(self, o, c_1, c_2):
        wt_1 = self.distance(o, c_2)**2 / (self.distance(o, c_2)**2 + self.distance(o, c_1)**2)
        wt_2 = 1 - wt_1
        
        return wt_1, wt_2
    
    def update_centroid(self, label, data_x, data_y):
        centroid_x = np.sum(np.multiply(np.power(label,2), data_x)) / np.sum(np.power(label,2))
        centroid_y = np.sum(np.multiply(np.power(label,2), data_y)) / np.sum(np.power(label,2))
        
        return [centroid_x, centroid_y]
    
    def compute_sse(self, centroids, data, labels):
        sse = 0
        for i in range(len(data)):
            wt = labels[i]
            sse_1 = wt[0]**2 * self.distance(data[i], centroids[0])**2
            sse_2 = wt[1]**2 * self.distance(data[i], centroids[1])**2
            sse = sse + sse_1 + sse_2
        return sse
    
    def fit(self, data):
        self.centroids = []
            
        # Initialize with first two data points
        for i in range(self.k):
            self.centroids.append(data[i])
        print("Initial Centroids: ", self.centroids)
        
        # Optimize in each iteration and update the centroids until hit the tolerance or max_iter
        for i in range(self.max_iter):
            self.labels = []
                
            for point in data:
                wt_1, wt_2 = self.compute_wt(point, self.centroids[0], self.centroids[1])
                self.labels.append([wt_1, wt_2])
            
            label_T = np.array(self.labels).transpose()
            data_x = data.transpose()[0]
            data_y = data.transpose()[1]
            prev_centroids = self.centroids[:]
            
            for k in range(self.k):
                # Update the new centroid
                self.centroids[k] = self.update_centroid(label_T[k], data_x, data_y)
            
            # Print the information of this iteration
            print("When iter=%i: " % i)
            print("The sum of SSE: ", self.compute_sse(self.centroids, data, self.labels))
            print ("The centroids: " , self.centroids)

            # Examine whether the change of centroids is too small to converge
            if (np.allclose(prev_centroids, self.centroids, atol=self.tol)):
                print("Converge and break!")
                break
            
            # Reach the max_iter steps
            if i == self.max_iter-1:
                print("Warning: Not converge!")
                    
    def predict(self,data):
        labels = []
        for point in data:
            wt_1, wt_2 = self.compute_wt(point, self.centroids[0], self.centroids[1])
            labels.append([wt_1, wt_2])
        return labels, self.centroids

def load_data():
    """ Load the .mat files from the Dataset

    Returns: 
        data: a collection of 2-d points
    """
    data = sio.loadmat("./data/dataset")["Points"]
    
    return data

def vis_cluster(data, labels, centroids):
    """
    Show the clustering results given original points and labels
    """    
    # get the distinct values of labels
    label_list = list(set(labels))

    # normalize the labels in order to map with the colormap
    norm = Normalize(vmin=0, vmax=len(label_list))
    
    # Plot points with different colors for different clusters
    for index in range(len(data)):
        plt.scatter(x=data[index][0],y=data[index][1], color=cm.jet(norm(labels[index])))

    for i in range(len(centroids)):
        plt.scatter(x=centroids[i][0],y=centroids[i][1], color='red', marker="s", s=50)

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

    startTime = time.time()

    # Biuld the fuzzy cluster with tol=1e-4 and max_iter = 400
    clustering = fuzzy_cluster(tol=1e-4, max_iter=400)
    clustering.fit(data)

    # Get the updated labels and centroids after training
    labels, centroids = clustering.predict(data)

    # Get the label id for each point
    labels_id = []
    for label in labels:
        labels_id.append(np.argmax(label))
    
    # Write to txt file
    path = "output_Fuzzy.txt"
    write_output(path, data, labels_id)

    print("The running time is: %f seconds" % (time.time()-startTime))

    # Visualize the clustering result
    vis_cluster(data, labels_id, centroids)
    plt.title('Fuzzy Clustering EM')
    plt.show()

if __name__ == "__main__":
    main()

