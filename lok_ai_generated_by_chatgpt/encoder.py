from sklearn.preprocessing import MultiLabelBinarizer

def encode_data(data, image_ids):
  encoder = MultiLabelBinarizer(classes=image_ids, sparse_output=False)
  return encoder.fit_transform(data)
