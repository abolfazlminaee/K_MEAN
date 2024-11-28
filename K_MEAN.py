import numpy as np
import random
def load_data(filename='points.txt'):
    data = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                values = line.strip().split(',')
                try:
                    data.append([float(value) for value in values])
                except ValueError:
                    continue 
    except FileNotFoundError:
        print(f"not found{filename} ")
    return np.array(data)
def preprocess_data(data):
    data = data[~np.isnan(data).any(axis=1)]
    return data
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def initialize_centroids(data, k):
    indices = random.sample(range(len(data)), k)
    return data[indices]
def assign_clusters(data, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
    return clusters
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters.values():
        if len(cluster) > 0:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(np.zeros(3))  
    return np.array(new_centroids)
def k_means_clustering(data, k):
    centroids = initialize_centroids(data, k)
    previous_centroids = centroids + 1  
    while not np.all(centroids == previous_centroids):
        previous_centroids = centroids.copy()
        clusters = assign_clusters(data, centroids)
        centroids = update_centroids(clusters)
    return clusters, centroids
k = int(input("please enter the number of clusters: "))
if k <= 0:
    print("It must be bigger than zero")
else:
    data = load_data('points.txt')
    data = preprocess_data(data)
    clusters, centroids = k_means_clustering(data, k)
    print("centers of clusters:")
    print(centroids)
    for cluster_id, points in clusters.items():
        print(f"cluster{cluster_id}:")
        for point in points:
            print(point)