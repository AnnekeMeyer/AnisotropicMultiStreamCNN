__author__ = 'gchlebus'

import numpy as np
import scipy.ndimage
import os
import SimpleITK as sitk
from collections import namedtuple
from utils import resampleToReference, makedir, binaryThresholdImage
from tqdm import tqdm

Case = namedtuple("Case", ["img_tra", "img_cor", "img_sag", "img_gt"])

DIR_SUFFIX = "_deformed"

def elastic_transform(images, sigma=4, control_points=3):
  """
  Elastic image transformation according to the 3D U-net paper by Cicek.
  :param images: List of images. Image is a 3 (x, y, channel) or 4 (x, y, z, channel) dimensional np.ndarray. The
  images should be of the same size.
  :param sigma: The deformation vectors at each control point location are drawn from normal distribution with 0
  mean and std dev given by elastic_sigma. Providing a tuple specifies sigma for each dimension.
  :param control_points: Count of control points of the deformation field. Providing a tuple specifies
  control_point_count for each dimension.
  :return: List of deformed images.
  """
  if not hasattr(sigma, '__len__'):
    sigma = (sigma,) * 3
  sigma = np.array(sigma)

  if not hasattr(control_points, '__len__'):
    control_points = (control_points,) * 3
  control_points = np.array(control_points)

  if not isinstance(images, list):
    images = [images]

  image_shape = images[0].shape

  if len(image_shape) == 3:  # 2D
    control_points = np.hstack((control_points[:2], np.array([1])))
    deformation_x = np.random.normal(0, sigma[0], control_points)
    deformation_y = np.random.normal(0, sigma[1], control_points)
    dc = np.zeros(control_points)
    dx = scipy.ndimage.interpolation.zoom(deformation_x, image_shape[:3] / control_points.astype(np.float32))
    dy = scipy.ndimage.interpolation.zoom(deformation_y, image_shape[:3] / control_points.astype(np.float32))
    dc = scipy.ndimage.interpolation.zoom(dc, image_shape[:3] / control_points.astype(np.float32))
    x, y, c = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), np.arange(image_shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(c + dc, (-1, 1))
  elif len(image_shape) == 4:  # 3D
    control_points = np.hstack((control_points, np.array([1])))
    deformation_x = np.random.normal(0, sigma[0], control_points)
    deformation_y = np.random.normal(0, sigma[1], control_points)
    deformation_z = np.random.normal(0, sigma[2], control_points)
    dc = np.zeros(control_points)
    dx = scipy.ndimage.interpolation.zoom(deformation_x, image_shape[:4] / control_points.astype(np.float32))
    dy = scipy.ndimage.interpolation.zoom(deformation_y, image_shape[:4] / control_points.astype(np.float32))
    dz = scipy.ndimage.interpolation.zoom(deformation_z, image_shape[:4] / control_points.astype(np.float32))
    dc = scipy.ndimage.interpolation.zoom(dc, image_shape[:4] / control_points.astype(np.float32))
    x, y, z, c = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), np.arange(image_shape[2]),
                             np.arange(image_shape[3]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1)), np.reshape(c + dc,
                                                                                                                (-1, 1))
  return list([
    scipy.ndimage.interpolation.map_coordinates(img, indices, mode='reflect', order=1).reshape(image_shape)
    for img in images
  ])


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="Create additional images by elastic deformation.")
  parser.add_argument("-n", "--num-iterations", default=4, help="How many times each of input datasets should be "
                                                                "deformed.", type=int)
  parser.add_argument("-i", "--input-dir", help="Input data directory.", type=str, default="")
  parser.add_argument("-cp", "--control-points", default=3, type=int)
  parser.add_argument("-s", "--sigma", default=4, type=float)
  return parser.parse_args()


def get_subdirectories(dir_path):
  if not dir_path and not os.path.exists(dir_path):
    print("Input directory not found: %s" % args.input_dir)
    exit(1)
  dirs = []
  for name in os.listdir(dir_path):
    fullPath = os.path.join(dir_path, name)
    if os.path.isdir(fullPath) and DIR_SUFFIX not in fullPath:
      dirs.append(fullPath)
  return dirs


def load_images(dir_path):
  return Case(
    img_tra=sitk.ReadImage(os.path.join(dir_path, "roi_tra.nrrd")),
    img_cor=sitk.ReadImage(os.path.join(dir_path, "roi_cor.nrrd")),
    img_sag=sitk.ReadImage(os.path.join(dir_path, "roi_sag.nrrd")),
    img_gt=sitk.ReadImage(os.path.join(dir_path, "roi_GT.nrrd"))
  )


def resample_to_gt(case):
  return Case(
    img_tra=resampleToReference(case.img_tra, case.img_gt, sitk.sitkLinear, 0),
    img_cor=resampleToReference(case.img_cor, case.img_gt, sitk.sitkLinear, 0),
    img_sag=resampleToReference(case.img_sag, case.img_gt, sitk.sitkLinear, 0),
    img_gt=case.img_gt
  )


def resample_to_origianal(case, original_case):
  return Case(
    img_tra=resampleToReference(case.img_tra, original_case.img_tra, sitk.sitkLinear, 0),
    img_cor=resampleToReference(case.img_cor, original_case.img_cor, sitk.sitkLinear, 0),
    img_sag=resampleToReference(case.img_sag, original_case.img_sag, sitk.sitkLinear, 0),
    img_gt=case.img_gt
  )


def save_case(case, dir_path):
  makedir(dir_path)
  sitk.WriteImage(case.img_tra, os.path.join(dir_path, "roi_tra.nrrd"))
  sitk.WriteImage(case.img_cor, os.path.join(dir_path, "roi_cor.nrrd"))
  sitk.WriteImage(case.img_sag, os.path.join(dir_path, "roi_sag.nrrd"))
  sitk.WriteImage(binaryThresholdImage(case.img_gt, 0.5), os.path.join(dir_path, "roi_GT.nrrd"))


def case_to_arraylist(case):
  return [
    sitk.GetArrayFromImage(case.img_tra),
    sitk.GetArrayFromImage(case.img_cor),
    sitk.GetArrayFromImage(case.img_sag),
    sitk.GetArrayFromImage(case.img_gt)
  ]


def arraylist_to_case(array_list, source_case):
  img_tra = sitk.GetImageFromArray(array_list[0])
  img_tra.CopyInformation(source_case.img_tra)

  img_cor = sitk.GetImageFromArray(array_list[1])
  img_cor.CopyInformation(source_case.img_cor)

  img_sag = sitk.GetImageFromArray(array_list[2])
  img_sag.CopyInformation(source_case.img_sag)

  img_gt = sitk.GetImageFromArray(array_list[3])
  img_gt.CopyInformation(source_case.img_gt)

  return Case(img_tra=img_tra, img_cor=img_cor, img_sag=img_sag, img_gt=img_gt)


if __name__ == "__main__":
  args = parse_args()

  for data_dir in tqdm(get_subdirectories(args.input_dir)):
    original_case = load_images(data_dir)
    resampled_case = resample_to_gt(original_case)
    images = case_to_arraylist(resampled_case)
    for i in range(args.num_iterations):
      deformed_images = elastic_transform(images, sigma=args.sigma,
                                          control_points=args.control_points)
      deformed_case = arraylist_to_case(deformed_images, resampled_case)
      deformed_case = resample_to_origianal(deformed_case, original_case)
      save_case(deformed_case, data_dir + DIR_SUFFIX + "%d" % i)
  print("Finished.")
