__author__ = 'gchlebus'

from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
#from keras import Callback
from keras import backend as K
import numpy as np
import os
from enum import Enum
import UNET3D_MultiStream_v2
from data_generation import DataGenerator
from keras.utils import plot_model
import csv
import tensorflow as tf

CSV_FILENAME = "log.csv"
MODEL_FILENAME = "model.h5"


class ModelType(Enum):
  SinglePlane = "SinglePlane"
  TriplePlane = "MultiPlane"
  DualPlane = "DualPlane"


# custom Callback that adds learning rate for each epoch to logfiles
class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        lr = K.eval(self.model.optimizer.lr)
        logs['LR'] = lr


def make_dirs(dir_path):
  try:
    os.makedirs(dir_path)
  except IOError:
    pass


def get_model(model_type, lr, dropout_rate=0, batch_normalization=False, upsampling_mode="transpose_conv", volumeSize_slices = 38,
              verbose=False):

  # from keras import backend as K
  # K.set_learning_phase(1)
  # set learning phase. bugfix for problems caused by workers that first run training without BN and then are assigned training with BN.
  #"learning phase" is a flag is a flag which indicates training/inference. It is set to 1 when using e.g. fit and to 0 when using e.g. predict.
  # K.set_learning_phase(False) sets the "learning phase" to be always 0, i.e. fit will have the model behave in inference mode (e.g. no dropout and
  # BatchNorm behavior set to inference).# '


  m = None
  if model_type == ModelType.SinglePlane:
    m = UNET3D_MultiStream_v2.get_net_singlePlane(dropout_rate=dropout_rate, batch_normalization=batch_normalization,
                                                  upsampling_mode=upsampling_mode, volumeSize_slices= volumeSize_slices)
  elif model_type == ModelType.TriplePlane:
    m = UNET3D_MultiStream_v2.get_net_triplePlane(dropout_rate=dropout_rate, batch_normalization=batch_normalization,
                                                  upsampling_mode=upsampling_mode, volumeSize_slices=volumeSize_slices)
  elif model_type == ModelType.DualPlane:
    m = UNET3D_MultiStream_v2.get_net_dualPlane(dropout_rate=dropout_rate, batch_normalization=batch_normalization,
                                                upsampling_mode=upsampling_mode, volumeSize_slices=volumeSize_slices)

  if m is not None:
    m.compile(optimizer=Adam(lr=lr), loss=UNET3D_MultiStream_v2.dice_coef_loss,
              metrics=[UNET3D_MultiStream_v2.dice_coef])
    if verbose:
      print(m.summary())

  else:
    raise RuntimeError("Invalid model type: %s", model_type)
  return m


def train(train_filenames, val_tra, val_cor, val_sag, val_GT, model_type, batch_size, epochs,
          csv_file, model_file, lr=5e-5, lr_scheduling=False, early_stop=False, initial_epoch=0, nr_augmentations =
          1, nr_elastic_deformations = 0, data_dir='/data/anneke/prostate-data/preprocessed/train/',
          tensorboard_log_name = 'tensorboard/', dropout_rate=0, batch_normalization=False,
          upsampling_mode="transpose_conv", volumeSize_slices=38):

  print('Batch Size Type (train): ',type(batch_size))
  # define Keras callbacks
  csv_logger = CSVLogger(csv_file, append=True, separator=';')
  earlyStopImprovement = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=1, mode='auto')
  LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=25, verbose=1, mode='auto', min_lr=1e-8,
                              epsilon=0.001)
  model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1)
  learningRateTracker = LearningRateTracker()
  tensorboard = TensorBoard(log_dir=tensorboard_log_name, histogram_freq=0, batch_size=4, write_graph=False,
                            write_grads=False, write_images=False)

  val_GT = val_GT.astype(np.uint8)
  print('----- clear session ----')
  K.clear_session()
  print('-' * 30)
  print('Creating and compiling model...')
  print('-' * 30)
  model = get_model(model_type, lr, dropout_rate, batch_normalization, upsampling_mode, volumeSize_slices)
  model.summary()  ## print model architecture
  # plot_model(model, to_file='model.png', show_shapes=True)

  if os.path.exists(model_file):
    print("Loading model from %s" % model_file)
    model.load_weights(model_file)

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  # learning rate tracker has to be inserted as callback before csv_logger
  cb = [learningRateTracker,csv_logger, model_checkpoint, tensorboard]
  if early_stop:
    cb.append(earlyStopImprovement)
  if lr_scheduling:
    cb.append(LRDecay)

  print('Callbacks: ', cb)


  # create validation data list according to number of input planes
  if model_type == ModelType.SinglePlane:
    val_data = [[val_tra], val_GT]
    n_planes = 1
  elif model_type == ModelType.DualPlane:
    val_data = [[val_tra, val_sag], val_GT]
    n_planes = 2
  elif model_type == ModelType.TriplePlane:
    val_data = [[val_tra, val_cor, val_sag], val_GT]
    n_planes = 3


  # data generator for online-learning
  params = {'batch_size': batch_size,
            'n_planes': n_planes,
            'shuffle': True,
            'data_dir': data_dir,
            'volumeSize_slices': volumeSize_slices}


  train_arr = train_filenames
  train_id_list = train_arr.tolist()
  augemntation_list=[]

  for i in range(1, int(nr_augmentations/(nr_elastic_deformations+1))):
    #print('in loop', nr_augmentations, int(nr_augmentations/(nr_elastic_deformations+1)))
    augemntation_list = augemntation_list + train_id_list
  train_id_list = train_id_list + augemntation_list

  # for elastic deformation, add e.g. '[dirname]_deformed0' to input list for data generator
  # as elastic deformation is not performed on-the-fly.
  if nr_elastic_deformations > 0:
    elastic_deformed_imgs_list = []
    for item in train_id_list:
      for i in range(0,nr_elastic_deformations):
        elastic_dir_name = item + '_deformed' + str(i)
        elastic_deformed_imgs_list.append(elastic_dir_name)# = elastic_deformed_imgs_list + elastic_dir_name
    train_id_list = train_id_list + elastic_deformed_imgs_list
  # append image_id_list nr_augmentation times to image_id_list

  print(train_id_list)


  #print(train_id_list)

  training_generator = DataGenerator(train_id_list, **params)

  print('train_id_list_length:', len(train_id_list))

  history = model.fit_generator(generator=training_generator,
                                validation_data=val_data,
                                use_multiprocessing=True, epochs=epochs, callbacks=cb,
                                workers=4, initial_epoch= initial_epoch)
  # print(history.history.keys())
  model.save(model_file[:-3]+'_final.h5')


  #tf.reset_default_graph()
  # TODO find out how to retrieve best validation score from the history
  return history


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="Start training.")
  parser.add_argument('-m', '--model-type', choices=ModelType._member_names_, default=ModelType.TriplePlane.name)
  parser.add_argument('-i', '--input-dir', help="path to input directory with image data")
  parser.add_argument('-lr', '--learning-rate', type=float, default=5e-5, help="learning rate (default %(default)d)")
  parser.add_argument('-bs', '--batch-size', type=int, default=1, help="batch size (default %(default)d)")
  parser.add_argument('-e', '--epochs', type=int, default=100, help="epoch count (default %(default)d)")
  parser.add_argument('--early-stop', action="store_true", help="use early stop")
  parser.add_argument('--lr-decay', action="store_true", help="use learning rate decay")
  parser.add_argument('output', help="output directory")
  return parser.parse_args()


def run_training(args):
  make_dirs(args.output)
  # get data
  csv_file = os.path.join(args.output, CSV_FILENAME)
  model_file = os.path.join(args.output, MODEL_FILENAME)


if __name__ == "__main__":
  args = parse_args()
  try:
    run_training(args)
  except RuntimeError as e:
    print("Error: %s" % e)
