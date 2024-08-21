import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import time

def generate_user_likes_data(num_users, num_images):
    """
    ユーザーの「いいね」データをカテゴリカルデータとして生成
    """
    np.random.seed(0)
    data = []
    for _ in range(num_users):
        # 各ユーザーが1から99のランダムな数の「いいね」を持つ
        num_likes = np.random.randint(1, 100)
        liked_images = np.random.choice([f'image-{i+1}' for i in range(num_images)], 
                                        size=num_likes, replace=False)
        data.append(liked_images)
    return data

def encode_data(data, num_images):
    """
    「いいね」データをOne-Hot Encodingでエンコード
    """
    mlb = MultiLabelBinarizer(classes=[f'image-{i+1}' for i in range(num_images)])
    return mlb.fit_transform(data)

def perform_clustering(train_data, n_clusters):
    """
    クラスタリングの実施
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(train_data)
    return kmeans

def adjust_clusters(kmeans, data, min_users_per_cluster, n_clusters):
    """
    各クラスタに最低人数を保証するように調整
    """
    clusters = kmeans.predict(data)
    cluster_sizes = np.bincount(clusters, minlength=n_clusters)

    # ユーザーを少ないクラスタに再割り当て
    for cluster_id in range(n_clusters):
        while cluster_sizes[cluster_id] < min_users_per_cluster:
            # 人数が少ないクラスタに追加
            for i in range(len(clusters)):
                if cluster_sizes[clusters[i]] > min_users_per_cluster:
                    cluster_sizes[clusters[i]] -= 1
                    clusters[i] = cluster_id
                    cluster_sizes[cluster_id] += 1
                    if cluster_sizes[cluster_id] >= min_users_per_cluster:
                        break
    return clusters

def print_cluster_summary(clusters, data, n_clusters):
    """
    各クラスタのユーザー「いいね」データの概要を表示
    """
    for cluster_id in range(n_clusters):
        cluster_data = data[clusters == cluster_id]
        print(f"\nクラスタ {cluster_id}:")
        print(f"  ユーザー数: {len(cluster_data)}")
        if len(cluster_data) > 0:
            # クラスタ内の「いいね」パターンの平均を表示
            cluster_mean = np.mean(cluster_data, axis=0)
            print(f"  平均「いいね」パターン: {cluster_mean}")

def recommend_images_per_cluster(clusters, data_encoded, n_clusters, num_recommendations=10):
    """
    各クラスタのおすすめ画像を出力
    """
    for cluster_id in range(n_clusters):
        cluster_data = data_encoded[clusters == cluster_id]
        if len(cluster_data) > 0:
            # クラスタ内の「いいね」パターンの平均を計算
            cluster_mean = np.mean(cluster_data, axis=0)
            # 最も「いいね」された画像のインデックスを取得
            recommended_images = np.argsort(cluster_mean)[-num_recommendations:][::-1]
            print(f"\nクラスタ {cluster_id} のおすすめ画像: {recommended_images}")
def recommend_images_for_new_user(kmeans, encoder, new_user_data, n_clusters, num_recommendations=10):
    """
    新しいユーザーに対して、おすすめ画像を返す
    """
    # 新しいユーザーの「いいね」データをOne-Hot Encodingでエンコード
    new_user_encoded = encoder.transform([new_user_data])
    
    # 新しいユーザーが属するクラスタを予測
    cluster = kmeans.predict(new_user_encoded)[0]
    
    # おすすめ画像を返す
    centroids = kmeans.cluster_centers_
    cluster_mean = centroids[cluster]
    
    # 最も「いいね」された画像のインデックスを取得
    recommended_images = np.argsort(cluster_mean)[-num_recommendations:][::-1]
    
    return recommended_images

def main():
    num_users = 10**3  # ユーザー数
    num_images = 10**3  # 画像数
    n_clusters = 10  # クラスタ数

    # クラスタあたりの最低ユーザー数を計算
    min_users_per_cluster = num_users // n_clusters // 10

    # ユーザーの「いいね」データの生成（1ユーザーあたりランダムな数の「いいね」）
    data = generate_user_likes_data(num_users, num_images)

    # データのOne-Hot Encoding
    encoder = MultiLabelBinarizer(classes=[f'image-{i+1}' for i in range(num_images)])
    data_encoded = encoder.fit_transform(data)

    # クラスタリングの実施
    start_time = time.time()
    kmeans = perform_clustering(data_encoded, n_clusters)
    end_time = time.time()
    print(f"クラスタリング時間: {end_time - start_time:.2f}秒")

    # クラスタの調整
    clusters = adjust_clusters(kmeans, data_encoded, min_users_per_cluster, n_clusters)

    # クラスタリング結果の表示
    print_cluster_summary(clusters, data_encoded, n_clusters)

    # 各クラスタのおすすめ画像を出力
    num_recommendations = 40
    recommend_images_per_cluster(kmeans, data_encoded, n_clusters, num_recommendations)

    # 新しいユーザーの「いいね」データを設定
    new_user_data = [f'image-{i+1}' for i in np.random.choice(range(num_images), size=np.random.randint(1, 100), replace=False)]

    # 新しいユーザーに対しておすすめ画像を取得
    recommended_images = recommend_images_for_new_user(kmeans, encoder, new_user_data, n_clusters, num_recommendations)
    print(f"\n新しいユーザーの「いいね」データ: {new_user_data}")
    print(f"新しいユーザーにおすすめの画像: {recommended_images}")

if __name__ == "__main__":
    main()
