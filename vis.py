import vis_util
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import EfficientNetB1
from meta import REPO_PATH
import matplotlib.pyplot as plt
import numpy as np

path = "\\result\\2018_03_16_P103_shPerk_bQ\\64bit\\model"
data_path = "\\data\\2018_03_16_P103_shPerk_bQ"
loaded_model = tf.keras.models.load_model(REPO_PATH + path)

set_name = "2018_03_16_P103_shPerk_bQ"

pretrained_net = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", input_shape=(224,224,3))
])

# load example image with a query
afb = plt.imread(REPO_PATH + data_path + "\\afb.png")
query = plt.imread(REPO_PATH + data_path + "\\query.png")
query = np.array(query)[:,:,1]
target = vis_util.get_bbn(query, loaded_model, pretrained_net)

# calculate retrievals with Hamming distance <= 1 from the target
retrievals = vis_util.retrievals(loaded_model, afb, target, pretrained_net)

plt.imshow(retrievals, cmap = "gray")
plt.show()