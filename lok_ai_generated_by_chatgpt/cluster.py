from sklearn.cluster import SpectralClustering


def predict_cluster(spectral: SpectralClustering, data):
  return spectral.fit_predict(data)

def perform_clustering(train_data, n_clusters):
  spectral = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=0)
  clusters = predict_cluster(spectral, train_data)
  return spectral, clusters
