import matplotlib.pyplot as plt
from train import tbh_train
from model.tbh import TBH
import tensorflow as tf
import numpy as np
from meta import REPO_PATH

#calculate hash code for a single patch
def get_bbn(patch, loaded_model, pretrained_net, IMAGE_SHAPE=224):
    sample = np.reshape(patch,(1,IMAGE_SHAPE,IMAGE_SHAPE,1))
    sample = np.repeat(sample,3,3)
    sample = np.expand_dims(pretrained_net.predict(sample).flatten(),0)
    fc1 = loaded_model.encoder.fc_1(sample)
    fc2 = loaded_model.encoder.fc_2_1(fc1)
    mean, logvar = tf.split(fc2, num_or_size_splits=2, axis=1)
    bbn = (tf.sign(mean)+1)/2
    return bbn
  

#show retrievals with Hamming distance <= 1
def retrievals(model, afb, target, pretrained_net, IMAGE_SHAPE=224, STEP=32):

  #calculate hash codes for every block
  codes = []
  for i in range(0,afb.shape[0]-IMAGE_SHAPE,STEP):
    row = []
    for j in range(0,afb.shape[1]-IMAGE_SHAPE,STEP):
      row.append(get_bbn(afb[i:i+IMAGE_SHAPE,j:j+IMAGE_SHAPE], model, pretrained_net, IMAGE_SHAPE))
    codes.append(row)
  codes = np.array(codes)
  print(codes.shape)

  #calculate distance to the query code
  distances = np.squeeze(np.sum(np.abs(codes[:,:,:,:]-target),axis=-1),axis=-1)
  length = len(distances[0])
  range_distances = np.where(np.sort(distances.ravel())>=2)[0][0]
  idxs = np.argsort(distances.ravel())[::-1][-range_distances:]
  rows, cols = idxs//length, idxs%length

  result = afb.copy()
  for row, col in zip(rows,cols):
    result[row*STEP,col*STEP:col*STEP+IMAGE_SHAPE] = 0.
    result[row*STEP+IMAGE_SHAPE,col*STEP:col*STEP+IMAGE_SHAPE] = 0.
    result[row*STEP:row*STEP+IMAGE_SHAPE,col*STEP+IMAGE_SHAPE] = 0.
    result[row*STEP:row*STEP+IMAGE_SHAPE,col*STEP] = 0.
  return result
