import SimpleITK as sitk
import numpy as np
import os
import utils
#import matplotlib.pyplot as plt

import random as rn
rn.seed(53)  # seed random number generator for reproducible augmentation

def binaryThresholdImage(img, lowerThreshold):

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded

def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0], parameter_values[1], parameter_values[2])) +
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in
            np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]


def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom;
        qv[1] = (r[0, 2] - r[2, 0]) / denom;
        qv[2] = (r[1, 0] - r[0, 1]) / denom;
    return qv


def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters, flip_hor,
                           interpolator=sitk.sitkLinear, default_intensity_value=0.0):

    '''
    Generate the resampled images based on the given transformations.
    Args:
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''


    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)

        if flip_hor:
            arr = sitk.GetArrayFromImage(original_image)
            arr = np.flip(arr, axis=2)
            flipped = sitk.GetImageFromArray(arr)
            flipped.CopyInformation(original_image)
            original_image = flipped

        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)

        return aug_image



def check_if_doubles(rand_vectors, new_vector):
    for vec in rand_vectors:
        if new_vector == vec:
            return True
        else:
            return False



def augmentImages(img_tra, img_sag, img_cor, img_GT):

    ###  vector of transformation paramaters where randomly values are picked from ###
    theta_Arr = [-0.349, -0.174533, -0.0872665, 0.0, 0.0872665, 0.174533, 0.349]
    translationsXY_Arr = [-10, -8,-6, -4, -2, 0.0, 2, 4, 6, 8, 10]  # in mm
    translationsZ_Arr = [-4, -3, -2, 0.0, 2, 3, 4]
    scale_Arr = [0.9, 0.95, 1.0, 1.05, 1.1]
    flip_Arr = [True, False]
    #smooth_Arr = [0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5]



    rand_vec = np.array(
        [rn.randint(0, 6), # theta for rotation along z-axis
         rn.randint(1, 10), # translation x
         rn.randint(1, 10), # translation y
         rn.randint(0, 6), # translation z
         rn.randint(0, 4), # scaling factor
         rn.randint(0, 1)], # flip horizontal
         dtype=int)



    #out_dir = '/data/anneke/data_augmented/' # = outputDirectory
    #nr_augm = nr_augmentations

    ### define reference volume for augmentation ###
    dimension = len(img_tra.GetSize())
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()


    # Set the augmenting transform's center so that rotation is around the image center.
    img_center = np.array(
    img_tra.TransformContinuousIndexToPhysicalPoint(np.array(img_tra.GetSize()) / 2.0))
    aug_transform.SetCenter(img_center)


    # TODO: check if doubles
    # define transformation parameters
    theta_x = 0.0
    theta_y = 0.0
    theta_z = theta_Arr[rand_vec[0]]
    transl_x = translationsXY_Arr[rand_vec[1]]
    transl_y = translationsXY_Arr[rand_vec[2]]
    transl_z = translationsZ_Arr[rand_vec[3]]
    scale = scale_Arr[rand_vec[4]]
    flip_hor = flip_Arr[rand_vec[5]]
    #smooth = smooth_Arr[rand_vec[6]]

    transformation_parameters_list = similarity3D_parameter_space_regular_sampling([theta_x], [theta_y], [theta_z],
                                                                                   [transl_x], [transl_y], [transl_z],
                                                                                   [scale])


    ### spatial tranformation of anatomical image
    reference_image = img_tra
    augm_tra = augment_images_spatial(img_tra, reference_image, sitk.AffineTransform(dimension),
                                     aug_transform, transformation_parameters_list, flip_hor,
                                     interpolator=sitk.sitkLinear, default_intensity_value=0.0)

    reference_image = img_sag
    augm_sag = augment_images_spatial(img_sag, reference_image, sitk.AffineTransform(dimension),
                                      aug_transform, transformation_parameters_list, flip_hor,
                                      interpolator=sitk.sitkLinear, default_intensity_value=0.0)

    reference_image = img_cor
    augm_cor = augment_images_spatial(img_cor, reference_image, sitk.AffineTransform(dimension),
                                      aug_transform, transformation_parameters_list, flip_hor,
                                      interpolator=sitk.sitkLinear, default_intensity_value=0.0)

    reference_image = img_GT
    augm_GT = augment_images_spatial(img_GT, reference_image, sitk.AffineTransform(dimension),
                                      aug_transform, transformation_parameters_list, flip_hor,
                                      interpolator=sitk.sitkNearestNeighbor, default_intensity_value=0.0)



    return augm_tra, augm_sag, augm_cor, augm_GT

