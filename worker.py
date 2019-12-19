__author__ = 'gchlebus'

import os
from hpbandster.core.worker import Worker
from utils import makedir
import pandas as pd
import train
import numpy as np

MODEL_FILENAME = "model.h5"
CSV_FILE = "log.csv"


class KerasWorker(Worker):
  def __init__(self, run_id, out_directory, nameserver=None, nameserver_port=None, logger=None, host=None, id=None,
               timeout=None, array_dir='',
               data_dir='',
               volumeSize_slices=36):
    Worker.__init__(self, run_id, nameserver, nameserver_port, logger, host, id, timeout)
    self.__out_directory = out_directory
    self.array_dir = array_dir
    self.data_dir = data_dir
    self.volumeSize_slices = volumeSize_slices

  def compute(self, config_id, config, budget, working_directory, nr_augmentations=10, nr_elastic_deformations = 4, fold_id = '1', modelType = train.ModelType.DualPlane):
    model_directory = os.path.join(self.__out_directory, "_".join([str(x) for x in config_id]))
    makedir(model_directory)
    stop_epoch = max(int(budget), 1)
    loss = self.train(stop_epoch, model_directory,  nr_augmentations=nr_augmentations,
                      nr_elastic_deformations = nr_elastic_deformations, fold_id = fold_id,  modelType = modelType, **config)
    return dict(
      loss=loss,
      info=loss  # not sure, whether info can be completely omitted
    )

  def train(self, stop_epoch, model_directory,  nr_augmentations=10, nr_elastic_deformations = 0, fold_id = '1', modelType = train.ModelType.DualPlane, **kwargs ):
    """
    Run a training session.
    :param stop_epoch: Stops training at the specified epoch.
    :param model_directory: The resulting model will be saved under this directory. If the model_directory already
    exists, then resume the training using the saved model.
    :param kwargs: Additional model/training hyperparameters.
    :return: Validation loss of the best validation.
    """
    # TODO Anneke: load images (should load exactly the same data across train calls)


    # load validation data

    # get image size
    inPlaneSize = np.load(self.array_dir + 'fold' + fold_id + '_val_imgs_tra.npy').shape[-2]
    print('inPlaneSize train.py', inPlaneSize)

    a = int((inPlaneSize - 4 * self.volumeSize_slices) / 2)
    b = int((inPlaneSize/4 - self.volumeSize_slices) / 2)
    val_imgs_tra = np.load(self.array_dir + 'fold' + fold_id + '_val_imgs_tra.npy')[:, b:-b, a:-a, a:-a, :]
    val_imgs_cor = np.load(self.array_dir + 'fold' + fold_id + '_val_imgs_cor.npy')[:, a:-a, b:-b, a:-a, :]
    val_imgs_sag = np.load(self.array_dir + 'fold' + fold_id + '_val_imgs_sag.npy')[:, a:-a, a:-a, b:-b, :]
    val_GT = np.load(self.array_dir + 'fold' + fold_id + '_val_GT.npy')[:, a:-a, a:-a, a:-a, :]

    # load training list
    train_filenames1 = np.load('Folds_ProstateX/train_fold' + fold_id + '.npy')
    print(train_filenames1.size)
    train_filenames2 = np.load('Folds_MD/train_fold' + fold_id + '.npy')

    train_filenames = np.concatenate([train_filenames1,train_filenames2], axis=0)


    csv_file = os.path.join(model_directory, CSV_FILE)
    model_file = os.path.join(model_directory, MODEL_FILENAME[:-3]+'.h5')
    #print(model_directory[:-6]+'tensorboard')
    tensorboard_filename = os.path.join(model_directory[:-6],'tensorboard', model_directory[-5:])

    print('tensorboard filename', tensorboard_filename)

    train.train(train_filenames, val_imgs_tra, val_imgs_cor, val_imgs_sag, val_GT, modelType,
                batch_size=kwargs.get("batch_size", 1),
                epochs=stop_epoch, initial_epoch=self.initial_epoch(csv_file),
                lr= self.learning_rate(csv_file, kwargs.get("learning_rate", 1e-5)), csv_file=csv_file, model_file=model_file,
                lr_scheduling=kwargs.get("lr_scheduling", True), early_stop=kwargs.get("early_stop", False),
                nr_augmentations=nr_augmentations, nr_elastic_deformations = nr_elastic_deformations, data_dir=self.data_dir, tensorboard_log_name = tensorboard_filename,
                dropout_rate=kwargs.get("dropout_rate", 0),
                batch_normalization=kwargs.get("batch_normalization", False),
                upsampling_mode=kwargs.get("upsampling_mode", "transpose_conv"),
                volumeSize_slices=self.volumeSize_slices)



    return self.best_val_loss(csv_file)


  def initial_epoch(self, csv_file):
    if not os.path.exists(csv_file):
      return 0
    try:
      df = pd.read_csv(csv_file, sep=';')
      return len(df["epoch"].tolist())
    except pd.errors.EmptyDataError:
      return 0

  def best_val_loss(self, csv_file):
    if not os.path.exists(csv_file):
      return 99999
    try:
      df = pd.read_csv(csv_file, sep=';')
      return df["val_loss"].min()
    except pd.errors.EmptyDataError:
      return 0

  def learning_rate(self, csv_file, initial_learning_rate):
    if not os.path.exists(csv_file):
      return initial_learning_rate
    try:
      df = pd.read_csv(csv_file, sep=';')
      return df["LR"][df.index[-1]]
    except pd.errors.EmptyDataError:
      return initial_learning_rate




#
# if __name__ == "__main__":
#
#   import sys
#
#   os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#   fold_id = sys.argv[2]
#
#   nr_elastic_deformations = 4
#
#   import ConfigSpace as cs
#   from example_configspace import get_configspace
#
#
#
#   ### single #######
#
#   worker = KerasWorker(run_id=str(4 + 1 * int(fold_id)), out_directory="train_single", volumeSize_slices=36,
#                         data_dir='data/preprocessed_imgs/',
#                        array_dir="data/arays/")
#   config_space = get_configspace()
#
#   config = config_space.sample_configuration().get_dictionary()
#
#   config = {
#     "batch_normalization": False,
#     "dropout_rate": 0.6,
#     "upsampling_mode": "upsampling",
#     "learning_rate": 0.0001282905140575413,
#     "early_stop": True
#   }
#
#   print(config)
#   res = worker.compute(config_id=(0, 0, int(fold_id)), config=config, budget=270, working_directory='.',
#                        nr_augmentations=10, nr_elastic_deformations=nr_elastic_deformations, fold_id=fold_id,
#                        modelType=train.ModelType.SinglePlane)
#
#




