import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import time

def generate_user_likes_data(num_users, num_images):
    """
    ユーザーの「いいね」データの生成
    """
    np.random.seed(0)
    data = np.random.randint(2, size=(num_users, num_images))
    return data

def encode_data(data):
    """
    「いいね」データのOneHot Encoding
    """
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(data)

def perform_clustering(train_data, n_clusters):
    """
    クラスタリングの実施
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(train_data)
    return kmeans

def classify_with_clusters(kmeans, test_data):
    """
    クラスタを用いて分類する
    """
    return kmeans.predict(test_data)

def evaluate_clustering(kmeans, data, labels):
    """
    クラスタリングの評価
    """
    distances = pairwise_distances_argmin_min(data, kmeans.cluster_centers_)
    accuracy = np.mean(labels == kmeans.predict(data))
    return accuracy

def print_cluster_summary(kmeans, data, n_clusters):
    """
    各クラスタのユーザー「いいね」データの概要を表示
    """
    clusters = kmeans.predict(data)
    for cluster_id in range(n_clusters):
        cluster_data = data[clusters == cluster_id]
        print(f"\nクラスタ {cluster_id}:")
        print(f"  ユーザー数: {len(cluster_data)}")
        if len(cluster_data) > 0:
            # クラスタ内の「いいね」パターンの平均を表示
            cluster_mean = np.mean(cluster_data, axis=0)
            print(f"  平均「いいね」パターン: {cluster_mean}")

def main():
    num_users = 10**3  # ユーザー数
    num_images = 10**3  # 画像数
    n_clusters = 10  # クラスタ数

    # ユーザーの「いいね」データの生成
    data = generate_user_likes_data(num_users, num_images)

    # データを60%と40%に分割
    split_index = int(0.6 * num_users)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # データのエンコード
    train_data_encoded = encode_data(train_data)
    test_data_encoded = encode_data(test_data)

    # クラスタリングの実施
    start_time = time.time()
    kmeans = perform_clustering(train_data_encoded, n_clusters)
    end_time = time.time()
    print(f"クラスタリング時間: {end_time - start_time:.2f}秒")

    # クラスタを用いてテストデータを分類
    test_labels = classify_with_clusters(kmeans, test_data_encoded)

    # クラスタリングの評価
    train_labels = kmeans.predict(train_data_encoded)
    train_accuracy = evaluate_clustering(kmeans, train_data_encoded, train_labels)
    test_accuracy = evaluate_clustering(kmeans, test_data_encoded, test_labels)

    print(f"サンプルデータのクラスタリング精度: {train_accuracy:.2f}")
    print(f"テストデータの分類精度: {test_accuracy:.2f}")

    # クラスタリング結果の表示
    print_cluster_summary(kmeans, train_data_encoded, n_clusters)

if __name__ == "__main__":
    main()
