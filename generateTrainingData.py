__author__ = 'ameyer'

import SimpleITK as sitk
import numpy as np
import os
import preprocessing
import utils
import shutil
import dataAugmentation

ISO_ROI_SIZE = [184, 184, 184]
# 136^3
SLICE_THICKNESS_FACTOR = 4
NR_AUGMENTATIONS = 5

def cropAndResampleVolume(inputDir):


    files = os.listdir(inputDir)
    for file in files:
        filename = os.path.join(inputDir, file)
        if os.path.isfile(filename):
            if 'tra.nrrd' in file:
                print(file)
                img_tra = sitk.ReadImage(filename)
            if 'cor.nrrd' in file:
                print(file)
                img_cor = sitk.ReadImage(filename)
            if 'sag.nrrd' in file:
                print(file)
                img_sag = sitk.ReadImage(filename)

    #  normalize image to values between 0 and 1 while cropping 1st and 99th percentile
    img_tra, img_cor, img_sag = utils.normalizeIntensitiesPercentile(img_tra, img_cor, img_sag)

    # choose a target coordinate system (in this case: upsampled transversal)
    tra_upsampled = utils.resampleImage(img_tra, [0.6, 0.6, 0.6], sitk.sitkLinear, 0)


    tra_upsampled = utils.changeSizeWithPadding(tra_upsampled, ISO_ROI_SIZE)

    # compute intersecting ROI of the three orthogonal images in that target coordinate system
    region_tra, region_cor, region_sag, start, size = preprocessing.getCroppedIsotropicImgs('out/', [0.6, 0.6, 0.6],
                                                                                            ISO_ROI_SIZE[0],
                                                                                            SLICE_THICKNESS_FACTOR,
                                                                                            img_tra, img_cor,
                                                                                            img_sag)

    #  extract ROI from that target coordinate system
    roi_upsampled = sitk.RegionOfInterest(tra_upsampled, [size[0], size[1], size[2]],
                                          [start[0], start[1], start[2]])

    #  extract anisotropic ROI from original tra input image
    roi_tra_target = utils.resampleImage(roi_upsampled, [0.6, 0.6, 2.4], sitk.sitkLinear, 0)
    ROI_tra = utils.resampleToReference(img_tra, roi_tra_target, sitk.sitkLinear, 0)

    # similarly, extract ansisotropic ROI for cor input image
    roi_cor_target = utils.resampleImage(roi_upsampled, [0.6, 2.4, 0.6], sitk.sitkLinear, 0)
    ROI_cor = utils.resampleToReference(img_cor, roi_cor_target, sitk.sitkLinear, 0)

    # similarly, extract ansisotropic ROI for sag input image
    roi_sag_target = utils.resampleImage(roi_upsampled, [2.4, 0.6, 0.6], sitk.sitkLinear, 0)
    ROI_sag = utils.resampleToReference(img_sag, roi_sag_target, sitk.sitkLinear, 0)

    return ROI_tra, ROI_cor, ROI_sag


def cropAndResampleVolumes(inputDir_imgs, input_dir_GT, output_dir):


    #  ---------- principle of preprocessing (obtaining ROI) ----------------
    # 1. choose a target coordinate system (in this case: upsampled transversal)
    # 2. compute intersecting ROI of the three orthogonal images in that target coordinate system
    # 3. extract ROI from that isotropic target coordinate system
    # 4. for each orthogonal image: resample ROI to the orthogonal-image-specific anisotropic resolution and map the original orthogonal image to that resampled ROI

    cases = os.listdir(input_dir_GT)
    img_index = 0

    for case in cases:


        # create output directory names
        print(case)
        img_dir = inputDir_imgs + '/'+ case + '/'
        GT_dir = input_dir_GT + '/' + case + '/'
        outDir = output_dir + '/preprocessed_imgs/' + case + '/'

        if not os.path.exists(outDir):
            os.mkdir(outDir)

        # read all necessary input volumes
        files = os.listdir(img_dir)
        for file in files:
            if os.path.isfile(img_dir + file):
                if 'upsampled' in file:
                    continue
                if 'tra.nrrd' in file:
                    print(file)
                    img_tra = sitk.ReadImage(img_dir + file)
                if 'cor.nrrd' in file:
                    print(file)
                    img_cor = sitk.ReadImage(img_dir + file)
                if 'sag.nrrd' in file:
                    print(file)
                    img_sag = sitk.ReadImage(img_dir + file)


        GT = sitk.ReadImage(GT_dir + '/prostate_smooth-label.nrrd')

        #  normalize image to values between 0 and 1 while cropping 1st and 99th percentile
        img_tra, img_cor, img_sag = utils.normalizeIntensitiesPercentile(img_tra, img_cor, img_sag)

        # choose a target coordinate system (in this case: upsampled transversal)
        tra_upsampled = utils.resampleImage(img_tra, [0.6, 0.6, 0.6], sitk.sitkLinear, 0)


        tra_upsampled = utils.changeSizeWithPadding(tra_upsampled, ISO_ROI_SIZE)

        # compute intersecting ROI of the three orthogonal images in that target coordinate system
        region_tra, region_cor, region_sag, start, size = preprocessing.getCroppedIsotropicImgs('out/', [0.6, 0.6, 0.6],
                                                                                                ISO_ROI_SIZE[0],
                                                                                                SLICE_THICKNESS_FACTOR,
                                                                                                img_tra, img_cor,
                                                                                                img_sag)

        #  extract ROI from that target coordinate system
        roi_upsampled = sitk.RegionOfInterest(tra_upsampled, [size[0], size[1], size[2]],
                                              [start[0], start[1], start[2]])

        #  extract anisotropic ROI from original tra input image
        roi_tra_target = utils.resampleImage(roi_upsampled, [0.6, 0.6, 2.4], sitk.sitkLinear, 0)
        ROI_tra = utils.resampleToReference(img_tra, roi_tra_target, sitk.sitkLinear, 0)

        # similarly, extract ansisotropic ROI for cor input image
        roi_cor_target = utils.resampleImage(roi_upsampled, [0.6, 2.4, 0.6], sitk.sitkLinear, 0)
        ROI_cor = utils.resampleToReference(img_cor, roi_cor_target, sitk.sitkLinear, 0)

        # similarly, extract ansisotropic ROI for sag input image
        roi_sag_target = utils.resampleImage(roi_upsampled, [2.4, 0.6, 0.6], sitk.sitkLinear, 0)
        ROI_sag = utils.resampleToReference(img_sag, roi_sag_target, sitk.sitkLinear, 0)

        # extract isotropic ROI for GT input image
        ROI_GT = utils.resampleToReference(GT, roi_upsampled, sitk.sitkNearestNeighbor, 0)

        utils.makeDirectory(outDir)

        # write preprocessed images
        sitk.WriteImage(ROI_tra, outDir + 'roi_tra.nrrd')
        sitk.WriteImage(ROI_cor, outDir + 'roi_cor.nrrd')
        sitk.WriteImage(ROI_sag, outDir + 'roi_sag.nrrd')
        sitk.WriteImage(ROI_GT, outDir + 'roi_GT.nrrd')





def generateFolds(directory, foldDir, nrSplits = 5):
    from sklearn.model_selection import KFold

    X = os.listdir(directory)
    X = [x for x in X if '_deformed' not in x]
    kf = KFold(n_splits=nrSplits, shuffle=True, random_state = 5)
    i=1
    for train, test in kf.split(X):
        train_data = np.array(X)[train]
        test_data = np.array(X)[test]
        print('train', train_data.shape, train_data)
        print('test', test_data.shape, test_data)
        np.save(foldDir+'/train_fold'+str(i), train_data)
        np.save(foldDir+'/test_fold' + str(i), test_data)
        i = i+1


def createArraysFromPatientList(list, input_directory):


    print(len(list),list)
    nr_data = len(list)
    img_arr_tra = np.zeros([nr_data, int(ISO_ROI_SIZE[0]/SLICE_THICKNESS_FACTOR), ISO_ROI_SIZE[0], ISO_ROI_SIZE[0], 1], dtype = np.float32)
    img_arr_cor = np.zeros(
        [nr_data,  ISO_ROI_SIZE[0], int(ISO_ROI_SIZE[0] / SLICE_THICKNESS_FACTOR), ISO_ROI_SIZE[0], 1],
        dtype=np.float32)
    img_arr_sag = np.zeros(
        [nr_data, ISO_ROI_SIZE[0], ISO_ROI_SIZE[0], int(ISO_ROI_SIZE[0] / SLICE_THICKNESS_FACTOR),  1],
        dtype=np.float32)
    gt_arr = np.zeros([nr_data, ISO_ROI_SIZE[0], ISO_ROI_SIZE[0], ISO_ROI_SIZE[0], 1], dtype = np.uint8)
    index=0

    for patient in list:

        img_tra = sitk.ReadImage(input_directory + '/' + patient + '/roi_tra.nrrd')
        img_cor = sitk.ReadImage(input_directory + '/' +patient + '/roi_cor.nrrd')
        img_sag = sitk.ReadImage(input_directory + '/' +patient + '/roi_sag.nrrd')
        gt = sitk.ReadImage(input_directory + '/' +patient + '/roi_GT.nrrd')

        print(patient)

        img_arr_tra[index, :, :, :, 0] = sitk.GetArrayFromImage(img_tra)
        img_arr_cor[index, :, :, :, 0] = sitk.GetArrayFromImage(img_cor)
        img_arr_sag[index, :, :, :, 0] = sitk.GetArrayFromImage(img_sag)
        gt_arr[index, :, :, :, 0] = sitk.GetArrayFromImage(gt)
        index = index+1

    return img_arr_tra, img_arr_cor, img_arr_sag, gt_arr


def createAnisotropicFoldArrays(data_directory, fold_dir, output_directory, nr_folds = 4):

    # patients = os.listdir(input_directory)
    for i in range(1, nr_folds+1):

        val_data = np.load(fold_dir+'/test_fold' + str(i) + '.npy')
        val_list = val_data.tolist()


        #img_arr_tra, img_arr_cor, img_arr_sag, gt_arr= createArraysFromPatientList(train_list, data_directory)
        val_img_arr_tra, val_img_arr_cor, val_img_arr_sag, val_gt_arr = createArraysFromPatientList(val_list, data_directory)


        #np.save(output_directory + 'fold' + str(i) + '_train_imgs_tra.npy', img_arr_tra)
        np.save(output_directory + '/fold' + str(i) + '_val_imgs_tra.npy', val_img_arr_tra)

        #np.save(output_directory + 'fold' + str(i) + '_train_imgs_cor.npy', img_arr_cor)
        np.save(output_directory + '/fold' + str(i) + '_val_imgs_cor.npy', val_img_arr_cor)

        #np.save(output_directory + 'fold' + str(i) + '_train_imgs_sag.npy', img_arr_sag)
        np.save(output_directory + '/fold' + str(i) + '_val_imgs_sag.npy', val_img_arr_sag)

        #np.save(output_directory + 'fold' + str(i) + '_train_GT.npy', gt_arr)
        np.save(output_directory + '/fold' + str(i) + '_val_GT.npy', val_gt_arr)


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="generate training data (preprocessing).")
  parser.add_argument('input_dir_imgs',  type=str, help="Directory with original input images (cor, sag, tra).")
  parser.add_argument('input_dir_GT', type=str, help="Directory with ground truth data.")
  parser.add_argument('output_dir', type=str, help="Output directory for preprocessed images.")
  parser.add_argument('nr_folds', type=int, help="Number of folds to be created.")
  return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()

    print('.... crop and resample volumes ....')
    cropAndResampleVolumes(args.input_dir_imgs, args.input_dir_GT, args.output_dir)

    print('.... split images to folds ....')
    utils.makeDirectory(os.path.join(args.output_dir, 'folds'))
    generateFolds(os.path.join(args.output_dir, 'preprocessed_imgs'),
                  os.path.join(args.output_dir, 'folds'), args.nr_folds)

    print('.... generate fold arrays ....')
    utils.makeDirectory(os.path.join(args.output_dir, 'arrays'))

    # function for fold array generation is only used for validation arrays
    createAnisotropicFoldArrays(os.path.join(args.output_dir, 'preprocessed_imgs'),
                                os.path.join(args.output_dir, 'folds'),
                                os.path.join(args.output_dir, 'arrays'), nr_folds = args.nr_folds)
