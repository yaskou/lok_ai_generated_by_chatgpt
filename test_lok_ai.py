import numpy as np
import uuid
from sklearn.model_selection import train_test_split

from lok_ai_generated_by_chatgpt.lok_ai import LokAi

def generate_user_likes_data(num_users, num_images):
  np.random.seed(0)

  image_ids = [str(uuid.uuid4()) for _ in range(num_images)]

  image_bias = np.random.rand(num_images)

  data = []
  for _ in range(num_users):
    num_likes = np.random.randint(1, 20)
    liked_images = np.random.choice(image_ids, size=num_likes, replace=False, p=image_bias / np.sum(image_bias))
    data.append(liked_images)

  return data, image_ids

def test_lok_ai():
  num_users = 10 ** 3
  num_images = 10 ** 4

  lok_ai = LokAi()

  data, image_ids = generate_user_likes_data(num_users, num_images)
  train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
  lok_ai.make_recommend(train_data, image_ids)

  for index, _ in enumerate(lok_ai.recommend_images_all_clusters):
    print(f"The recommends of cluster{index}: {lok_ai.recommend_images_all_clusters[index]}")

  recommends = lok_ai.recommend_for_user(test_data, image_ids)

  for index, customized_recommends in enumerate(recommends):
    print(f"The recommends of user{index}: {customized_recommends}")

if __name__ == "__main__":
  test_lok_ai()
