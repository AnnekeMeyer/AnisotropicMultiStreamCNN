__author__ = 'ameyer'

import os
import train
import numpy as np
import SimpleITK as sitk

MODEL_FILENAME = "model.h5"
volumeSize_slices = 46


def preprocess(inputImgDir, modelType=train.ModelType.TriplePlane):
  img_tra = sitk.ReadImage(os.path.join(inputImgDir, 'roi_tra.nrrd'))
  inPlaneSize = img_tra.GetSize()[0]

  a = int((inPlaneSize - 4 * volumeSize_slices) / 2)
  b = int((inPlaneSize / 4 - volumeSize_slices) / 2)

  arr_tra = np.zeros([1, volumeSize_slices, 4 * volumeSize_slices, 4 * volumeSize_slices, 1])
  print(arr_tra.shape)
  arr_tra[0, :, :, :, 0] = sitk.GetArrayFromImage(img_tra)  # [b:-b, a:-a, a:-a]

  img_sag = sitk.ReadImage(os.path.join(inputImgDir, 'roi_sag.nrrd'))
  arr_sag = np.zeros([1, 4 * volumeSize_slices, 4 * volumeSize_slices, volumeSize_slices, 1])
  arr_sag[0, :, :, :, 0] = sitk.GetArrayFromImage(img_sag)  # [ a:-a, a:-a, b:-b]

  img_cor = sitk.ReadImage(os.path.join(inputImgDir, 'roi_cor.nrrd'))
  arr_cor = np.zeros([1, 4 * volumeSize_slices, volumeSize_slices, 4 * volumeSize_slices, 1])
  arr_cor[0, :, :, :, 0] = sitk.GetArrayFromImage(img_cor)  # [a:-a, b:-b, a:-a]

  if modelType == train.ModelType.SinglePlane:
    return [arr_tra]
  if modelType == train.ModelType.DualPlane:
    return [arr_tra, arr_sag]
  if modelType == train.ModelType.TriplePlane:
    return [arr_tra, arr_cor, arr_sag]


def predict(image_dir, modelType, modelDirectory, **kwargs):
  model = train.get_model(modelType, lr=kwargs.get("learning_rate", 1e-5),
                          dropout_rate=kwargs.get("dropout_rate", 0),
                          batch_normalization=kwargs.get("batch_normalization", False),
                          upsampling_mode=kwargs.get("upsampling_mode", "transpose_conv"),
                          volumeSize_slices=volumeSize_slices)
  model.load_weights(os.path.join(modelDirectory, MODEL_FILENAME))

  inputs = preprocess(image_dir, modelType)
  out = model.predict(inputs, batch_size=1, verbose=1)
  return out


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="Inference.")
  parser.add_argument('model_type', choices=["single", "dual", "triple"], help="Path to the config space definition.")
  parser.add_argument('model_dir', type=str, help="Model directory.")
  parser.add_argument('image_dir', type=str, help="Input image directory.")
  parser.add_argument('output', type=str, help='Output filename')
  return parser.parse_args()


CONFIG = {
  "single": {
    "batch_normalization": False,
    "dropout_rate": 0.6,
    "upsampling_mode": "upsampling",
    "learning_rate": 0.0001282905140575413
  },
  "dual": {
    "batch_normalization": False,
    "dropout_rate": 0.2,
    "upsampling_mode": "transpose_conv",
    "learning_rate": 0.00013147776478275702
  },
  "triple": {
    "batch_normalization": True,
    "dropout_rate": 0.2,
    "upsampling_mode": "transpose_conv",
    "learning_rate": 0.00029905363591026105
  }
}

MODEL_TYPE = {
  "single": train.ModelType.SinglePlane,
  "dual": train.ModelType.DualPlane,
  "triple": train.ModelType.TriplePlane
}

if __name__ == "__main__":
  args = parse_args()

  config = CONFIG[args.model_type]
  modelType = MODEL_TYPE[args.model_type]
  out = predict(args.image_dir, modelType, args.model_dir, **config)
  pred = sitk.GetImageFromArray(out[0, :, :, :, 0])
  # pred.CopyInformation(image)
  sitk.WriteImage(pred, args.output)
