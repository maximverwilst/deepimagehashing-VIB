import tensorflow as tf
import os
import numpy as np
from meta import REPO_PATH
from util.data.set_processor import SET_DIM, SET_LABEL, SET_SPLIT, SET_SIZE


class ParsedRecord(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'cifar10')
        self.part_name = kwargs.get('part_name', 'train')
        self.batch_size = kwargs.get('batch_size', 256)
        rand = kwargs.get("random", True)
        self.data = self._load_data(rand)

    def _load_data(self, rand):
        def data_parser(tf_example: tf.train.Example):
            feat_dict = {'id': tf.io.FixedLenFeature([], tf.int64),
                         'feat': tf.io.FixedLenFeature([SET_DIM.get(self.set_name, 4096)], tf.float32),
                         'label': tf.io.FixedLenFeature([SET_LABEL.get(self.set_name, 10)], tf.float32)}
            features = tf.io.parse_single_example(tf_example, features=feat_dict)

            _id = tf.cast(features['id'], tf.int32)
            _feat = tf.cast(features['feat'], tf.float32)
            _label = tf.cast(features['label'], tf.int32)
            return _id, _feat, _label

        if self.set_name=="NETosis" and self.part_name=="train":
          strings = []
          for i in range(4):
            record_name = os.path.join(REPO_PATH, 'data', self.set_name, self.part_name + str(i)+ '.tfrecords')
            strings.append(record_name)
          
          tf.constant(strings, dtype = tf.string)
        else:
          record_name = os.path.join(REPO_PATH, 'data', self.set_name, self.part_name + '.tfrecords')
        data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).prefetch(self.batch_size)
        if rand:
          data = data.cache().repeat().shuffle(10000).batch(self.batch_size)
        else:
          data = data.cache().repeat().batch(self.batch_size)

        return data

    @property
    def output_contents(self):
        return ['fid', 'feature', 'label']


class Dataset(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'cifar10')
        self.batch_size = kwargs.get('batch_size', 256)
        self.code_length = kwargs.get('code_length', 32)
        self.rand = kwargs.get('random', True)
        self._load_data(self.rand)
        set_size = SET_SIZE.get(self.set_name)
        self.train_code = np.zeros([set_size[0], self.code_length])
        self.test_code = np.zeros([set_size[1], self.code_length])
        self.train_label = np.zeros([set_size[0], SET_LABEL.get(self.set_name, 10)])
        self.test_label = np.zeros([set_size[1], SET_LABEL.get(self.set_name, 10)])

    def _load_data(self, rand):
        # 1. training data
        settings = {'set_name': self.set_name,
                    'batch_size': self.batch_size,
                    'part_name': SET_SPLIT[0],
                    "random": rand}
        self.train_data = ParsedRecord(**settings).data

        # 2. test data
        settings['part_name'] = SET_SPLIT[1]
        self.test_data = ParsedRecord(**settings).data

    def update(self, entry, code, label, split):
        if split == SET_SPLIT[0]:
            self.train_code[entry, :] = code
            self.train_label[entry, :] = label
        elif split == SET_SPLIT[1]:
            self.test_code[entry, :] = code
            self.test_label[entry, :] = label
        else:
            raise NotImplementedError
