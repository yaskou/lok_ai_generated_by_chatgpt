import numpy as np


def recommend_per_cluster(clusters, data_encoded, n_clusters, num_recommendations=10):
  recommendations = {}
  for cluster_id in range(n_clusters):
    cluster_data = data_encoded[clusters == cluster_id]
    if len(cluster_data) > 0:
      cluster_mean = np.mean(cluster_data, axis=0)
      recommended_images = np.argsort(cluster_mean)[-num_recommendations:][::-1]
      recommendations[cluster_id] = recommended_images.tolist()
  return recommendations
