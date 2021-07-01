# deepimagehashing-VIB

This repository provides the implementation on the microscopy dataset from VIB. It largely follows the [base repository](https://github.com/maximverwilst/deepimagehashing-VAE).

## Requirements
```angular2
python=3.6
tensorflow>=2.5
tensorhub
numpy
matplotlib
```

## Data
This work supports `tf.data.TFRecordDataset` as the data feed. 
We demonstrate our implementation on a subset of the microscopy dataset:
* 2018_03_16_P103_shPerk_bQ ([Training](https://drive.google.com/file/d/1zxzUlab0NxMSIQ8H3428ox07pUIItjEt/view?usp=sharing), [Test](https://drive.google.com/file/d/1qnbg5KkB3yDQT-cSHmnZuL0UifpU9KJG/view?usp=sharing))

For other datasets, please refer to [`util/data/make_data.py`](./util/data/make_data.py) to build TFRecords.

Please organize the data folder as follows:
```angular2
data
  |-2018_03_16_P103_shPerk_bQ (or other dataset names)
    |-train.tfrecords
    |-test.tfrecords
```

Simply run
```angular2
python ./run_tbh.py
```
to train the model.

The resulting checkpoints will be placed in `./result/set_name/model/date_of_today` with tensorboard events in `./result/set_name/log/date_of_today`.

The mAP results shown on tensorboard are just for illustration (the actual score would be slightly higher than the ones on tensorboard), 
since I do not update all dataset codes upon testing. Please kindly evaluate the results by saving the proceeded codes after training.


The visualisations can be recreated by running
```angular2
python ./vis.py
```
