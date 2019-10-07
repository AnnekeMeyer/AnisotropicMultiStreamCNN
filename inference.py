__author__ = 'ameyer'

import os
from hpbandster.core.worker import Worker
from utils import makedir
#import pandas as pd
import train
import numpy as np
import SimpleITK as sitk

MODEL_FILENAME = "model.h5"
CSV_FILE = "log.csv"

volumeSize_slices = 46


# class KerasWorker(Worker):
#   def __init__(self, run_id, out_directory, nameserver=None, nameserver_port=None, logger=None, host=None, id=None,
#                timeout=None, array_dir='/data/anneke/prostate-data/whole-prostate-arrays/',
#                data_dir='/data/anneke/prostate-data/preprocessed/train/',
#                volumeSize_slices=36):
#     Worker.__init__(self, run_id, nameserver, nameserver_port, logger, host, id, timeout)
#     self.__out_directory = out_directory
#     self.array_dir = array_dir
#     self.data_dir = data_dir
#     self.volumeSize_slices = volumeSize_slices


def preprocess(inputImgDir, modelType= train.ModelType.MultiPlane):

  # load input imgs and create arrays



  img_tra = sitk.ReadImage(os.path.join(inputImgDir, 'roi_tra.nrrd'))
  inPlaneSize = img_tra.GetSize()[0]

  a = int((inPlaneSize - 4 * volumeSize_slices) / 2)
  b = int((inPlaneSize / 4 - volumeSize_slices) / 2)


  arr_tra = np.zeros([1, volumeSize_slices, 4 * volumeSize_slices, 4 * volumeSize_slices, 1])
  print(arr_tra.shape)
  arr_tra[0,:,:,:,0] = sitk.GetArrayFromImage(img_tra)#[b:-b, a:-a, a:-a]

  img_sag = sitk.ReadImage(os.path.join(inputImgDir, 'roi_sag.nrrd'))
  arr_sag = np.zeros([1,4 * volumeSize_slices, 4 * volumeSize_slices, volumeSize_slices,1])
  arr_sag[0, :, :, :, 0] = sitk.GetArrayFromImage(img_sag)#[ a:-a, a:-a, b:-b]

  img_cor = sitk.ReadImage(os.path.join(inputImgDir, 'roi_cor.nrrd'))
  arr_cor = np.zeros([1,4 * volumeSize_slices, volumeSize_slices, 4 * volumeSize_slices, 1])
  arr_cor[0,:, :, :, 0] = sitk.GetArrayFromImage(img_cor)#[a:-a, b:-b, a:-a]


  if modelType == train.ModelType.SinglePlane:
    return [arr_tra]
  if modelType == train.ModelType.DualPlane:
    return [arr_tra, arr_sag]
  if modelType == train.ModelType.MultiPlane:
    return [arr_tra, arr_cor, arr_sag]


def predict(inputDir, model_directory, out_directory,  modelType= train.ModelType.MultiPlane, **kwargs):

  model = train.get_model(modelType, lr=kwargs.get("learning_rate", 1e-5),
                          dropout_rate=kwargs.get("dropout_rate", 0),
                          batch_normalization=kwargs.get("batch_normalization", False),
                          upsampling_mode=kwargs.get("upsampling_mode", "transpose_conv"),
                          volumeSize_slices=volumeSize_slices)
  model.load_weights(os.path.join(model_directory, MODEL_FILENAME))

  cases = os.listdir(inputDir)
  for case in cases:
    inputs = preprocess(os.path.join(inputDir,case), modelType)

    out = model.predict(inputs, batch_size=1, verbose=1)
    print(out.shape)

    makedir(out_directory)
    pred = sitk.GetImageFromArray(out[0,:,:,:,0])
    roi_GT = sitk.ReadImage(os.path.join(inputDir,case, 'roi_GT.nrrd'))
    pred.CopyInformation(roi_GT)
    sitk.WriteImage(pred, os.path.join(out_directory,'predicted_'+case+'.nrrd'))

  return 'done'



if __name__ == "__main__":


  os.environ["CUDA_VISIBLE_DEVICES"] = '3'

  import ConfigSpace as cs
  from example_configspace import get_configspace

  dataset= 'Magdeburg'

 #### prediction #####

  for fold_id in range(1,5):
  #
    input_dir = os.path.join('/data/anneke/prostate-data',dataset, 'preprocessed/test/')

    config_space = get_configspace()
    config = config_space.sample_configuration().get_dictionary()

    confid_id = '0_0_' + str(fold_id)


    ### single ###
    print('single fold', fold_id)
    config = {
      "batch_normalization": False,
      "dropout_rate": 0.6,
      "upsampling_mode": "upsampling",
      "learning_rate": 0.0001282905140575413
    }
    modelType = train.ModelType.SinglePlane

    out_dir = os.path.join('prediction_single_ProstateX',dataset, confid_id)
    print(out_dir)
    model_directory = os.path.join('train_single_ProstateX', confid_id)

    predict(input_dir, model_directory, out_dir, modelType, **config)


    ### dual ###
    config = {
      "batch_normalization": False,
      "dropout_rate": 0.2,
      "upsampling_mode" : "transpose_conv",
      "learning_rate" : 0.00013147776478275702
    }
    modelType = train.ModelType.DualPlane
    print('dual fold', fold_id)
    out_dir = os.path.join('prediction_dual_ProstateX/',dataset, confid_id)
    model_directory = os.path.join('train_dual_ProstateX', confid_id)

    predict(input_dir, model_directory, out_dir, modelType, **config)

    ### multi-plane ###

    config = {
      "batch_normalization": True,
      "dropout_rate": 0.2,
      "upsampling_mode": "transpose_conv",
      "learning_rate": 0.00029905363591026105
    }

    modelType = train.ModelType.MultiPlane

    out_dir = os.path.join('prediction_multi_ProstateX/',dataset, confid_id)
    model_directory = os.path.join('train_multi_ProstateX', confid_id)

    predict(input_dir, model_directory, out_dir,  modelType, **config)




