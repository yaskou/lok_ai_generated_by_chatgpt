from lok_ai_generated_by_chatgpt.encoder import encode_data
from lok_ai_generated_by_chatgpt.cluster import perform_clustering, predict_cluster
from lok_ai_generated_by_chatgpt.recommend import recommend_per_cluster


class LokAi():
  def make_recommend(self, data, image_ids, max_num_recommendations = 50):
    data_encoded = encode_data(data, image_ids)
    n_clusters = int(data_encoded.shape[0] / 10)

    self.spectral, clusters = perform_clustering(data_encoded, n_clusters)
    self.recommend_images_all_clusters = recommend_per_cluster(clusters, data_encoded, n_clusters, max_num_recommendations)

  def recommend_for_user(self, data, image_ids, num_recommendations = 10):
    data_encoded = encode_data(data, image_ids)
    clusters = predict_cluster(self.spectral, data_encoded)
    return [self.recommend_images_all_clusters[cluster][:num_recommendations] for cluster in clusters]
