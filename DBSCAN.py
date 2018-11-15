import time
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from matplotlib.colors import Normalize

def load_data():
    """ Load the .mat files from the Dataset

    Returns: 
        data: a collection of 2-d points
    """
    data = sio.loadmat("./data/dataset")["Points"]
    
    return data

def vis_cluster(data, labels):
    """
    Show the clustering results given original points and labels
    """
    # get the distinct values of labels
    label_list = list(set(labels))

    # normalize the labels in order to map with the colormap
    norm = Normalize(vmin=0, vmax=len(label_list))

    # Plot points with different colors for different clusters, and black "X" points for outliers
    for index in range(len(data)):
        if labels[index] != -1:
            plt.scatter(x=data[index][0],y=data[index][1], color=cm.jet(norm(labels[index])))
        else:
            plt.scatter(x=data[index][0],y=data[index][1], color='black', marker = "X")

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

    # Biuld the DBSCAN cluster and fit the data
    clustering = DBSCAN(eps=0.12, min_samples=3).fit(data)
    
    # Get the labels for each point
    labels = clustering.labels_

    # Write to txt file
    path = "output_DBSCAN.txt"
    write_output(path, data, labels)

    print("The running time is: %f seconds" % (time.time()-startTime))
    
    # visualize the clustered points with matplotlib
    vis_cluster(data, labels)
    plt.title('DBSCAN Clustering')
    plt.show()

if __name__ == "__main__":
    main()