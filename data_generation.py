__author__ = 'ameyer'

import keras
import numpy as np
import dataAugmentation
import SimpleITK as sitk
import utils

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=2, n_planes=3,
                 shuffle=True, data_dir='',
                 volumeSize_slices=38):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_planes = n_planes
        self.shuffle = shuffle
        self.n_channels = 1
        self.data_dir = data_dir
        self.volumeSize_slices = volumeSize_slices
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_tra = np.empty([self.batch_size, self.volumeSize_slices, 4*self.volumeSize_slices, 4*self.volumeSize_slices,self.n_channels])
        X_sag = np.empty([self.batch_size, 4*self.volumeSize_slices,4*self.volumeSize_slices,self.volumeSize_slices, self.n_channels])
        X_cor = np.empty([self.batch_size, 4*self.volumeSize_slices,self.volumeSize_slices,4*self.volumeSize_slices, self.n_channels])
        Y = np.empty([self.batch_size, 4*self.volumeSize_slices,4*self.volumeSize_slices,4*self.volumeSize_slices,1], dtype=np.uint8)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            roi_tra = sitk.ReadImage(self.data_dir + '/' + ID + '/roi_tra.nrrd')
            roi_cor = sitk.ReadImage(self.data_dir + '/' + ID + '/roi_cor.nrrd')
            roi_sag = sitk.ReadImage(self.data_dir + '/' + ID + '/roi_sag.nrrd')
            roi_GT = sitk.ReadImage(self.data_dir + '/' + ID + '/roi_GT.nrrd')

            # augment sample
            augm_tra, augm_sag, augm_cor, augm_GT = dataAugmentation.augmentImages(roi_tra, roi_sag, roi_cor, roi_GT)

            # crop ROIS to input size of network
            # get size of uncropped ROI
            inPlaneSize = roi_tra.GetSize()[0]
            a = int((inPlaneSize - 4*self.volumeSize_slices) / 2)
            b = int((inPlaneSize/4 - self.volumeSize_slices) / 2)
            augm_tra = utils.cropImage(augm_tra, [a, a, b], [a, a, b])
            augm_cor = utils.cropImage(augm_cor, [a, b, a], [a, b, a])
            augm_sag = utils.cropImage(augm_sag, [b, a, a], [b, a, a])
            augm_GT = utils.cropImage(augm_GT, [a, a, a], [a, a, a])


            # store augmented sample
            X_tra[i, :, :, :, 0] = sitk.GetArrayFromImage(augm_tra)
            X_cor[i, :, :, :, 0] = sitk.GetArrayFromImage(augm_cor)
            X_sag[i, :, :, :, 0] = sitk.GetArrayFromImage(augm_sag)
            Y[i, :, :, :, 0] = sitk.GetArrayFromImage(augm_GT)


        return [X_tra, X_cor,X_sag] , [Y]

    def __len__(self):
        'Denotes the number of batches per epoch'
        print( 'List_ID', type(self.list_IDs))
        print('batch', type(self.batch_size))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [[X_tra, X_cor, X_sag], Y ] = self.__data_generation(list_IDs_temp)

        if self.n_planes == 1:
            return [X_tra], [Y]
        elif self.n_planes==2:
            return [X_tra, X_sag], [Y]
        elif self.n_planes==3:
            return [X_tra, X_cor,X_sag] , [Y]

