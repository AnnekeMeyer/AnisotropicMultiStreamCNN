from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, SpatialDropout3D, \
  BatchNormalization, Activation

smooth = 1.

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)


def repeatedConv3D(input, frequency, nrFilter, filterSize, strides):
  up = input
  for i in range(0, frequency):
    # up = UpSampling3D(size=strides)(up)
    up = Conv3DTranspose(nrFilter, filterSize, strides=strides, activation='relu', padding='same')(up)

  return up


def add_dropout(input, dropout_rate):
  if dropout_rate == 0:
    return input
  return SpatialDropout3D(rate=dropout_rate)(input)


def conv3d(input, filters, batch_normalization=False):

  out = Conv3D(int(filters), (3, 3, 3), padding="same", use_bias=not batch_normalization)(input)
  if batch_normalization:
    out = BatchNormalization()(out)
  return Activation("relu")(out)


def conv3d_transpose(input, filters, batch_normalization=False, filter_size=(2, 2, 2), strides=(2, 2, 2)):
  out = input
  out = Conv3DTranspose(filters, filter_size, strides=strides, padding="same", use_bias=not batch_normalization)(out)
  if batch_normalization:
    out = BatchNormalization()(out)
  return Activation("relu")(out)

def upsample(input, mode, filters, batch_normalization=False, repeat=1, filter_size=(2, 2, 2), strides=(2, 2, 2)):
  out = input
  for _ in range(repeat):
    if mode == "transpose_conv":
      out = conv3d_transpose(out, filters, batch_normalization, filter_size, strides)
    elif mode == "upsampling":
      out = UpSampling3D(size=strides)(out)
    else:
      raise RuntimeError("Not supported upsample mode %s" % mode)
  return out

def analysis_path(levels, input_shape, batch_normalization=False, up_strides=(2, 1, 1), pool_size=(1, 2, 2),
                  upsampling_mode="transpose_conv", filterFactor=1, base_filters=8):
  ret = dict()
  inputs = Input(input_shape + (1,))
  ret["inputs"] = inputs
  out = inputs
  for i in range(levels):
    filters = int(filterFactor * base_filters * 2 ** i)
    out = conv3d(out, filters, batch_normalization)
    out = conv3d(out, 2 * filters, batch_normalization)
    ret["up%d" % (i + 1)] = upsample(out, upsampling_mode, 2*filters, batch_normalization, repeat=2-i, strides=up_strides)
    out = MaxPooling3D(pool_size=pool_size)(out)
  ret["last_pool"] = out
  return ret


def synthesis_path(input, concat1, concat2, concat3, filterFactor=1, dropout_rate=0, batch_normalization=False,
                   upsampling_mode="transpose_conv"):
  up6 = upsample(input, upsampling_mode, int(128 * filterFactor), batch_normalization)
  up6 = concatenate([up6] + concat3)
  conv6 = conv3d(up6, int(64 * filterFactor), batch_normalization)
  conv6 = add_dropout(conv6, dropout_rate)
  conv6 = conv3d(conv6, int(64 * filterFactor), batch_normalization)

  up7 = upsample(conv6, upsampling_mode,  int(64 * filterFactor), batch_normalization)
  up7 = concatenate([up7] + concat2)
  conv7 = conv3d(up7,  int(32 * filterFactor), batch_normalization)
  conv7 = add_dropout(conv7, dropout_rate)
  conv7 = conv3d(conv7,  int(32 * filterFactor), batch_normalization)

  up8 = upsample(conv7, upsampling_mode, int(32 * filterFactor), batch_normalization)
  up8 = concatenate([up8] + concat1)
  conv8 = conv3d(up8, int(16 * filterFactor), batch_normalization)
  conv8 = add_dropout(conv8, dropout_rate)
  conv8 = conv3d(conv8, int(16 * filterFactor), batch_normalization)

  return Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)


def get_net_singlePlane(filterFactor=1, dropout_rate=0, batch_normalization=False, upsampling_mode="transpose_conv",
                        volumeSize_slices=38):
  analysis_tra = analysis_path(2, (volumeSize_slices, 4*volumeSize_slices, 4*volumeSize_slices), batch_normalization,
                               up_strides=(2, 1, 1),
                               pool_size=(1, 2, 2), upsampling_mode=upsampling_mode)

  conv3 = conv3d(analysis_tra["last_pool"], int(32 * filterFactor), batch_normalization)
  conv3 = conv3d(conv3, int(64 * filterFactor), batch_normalization)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = conv3d(pool3, int(128 * filterFactor), batch_normalization)
  conv4 = add_dropout(conv4, dropout_rate)
  conv4 = conv3d(conv4, int(128 * filterFactor), batch_normalization)

  out = synthesis_path(conv4,
                       concat3=[conv3],
                       concat2=[analysis_tra["up2"]],
                       concat1=[analysis_tra["up1"]],
                       filterFactor=filterFactor,
                       dropout_rate=dropout_rate,
                       batch_normalization=batch_normalization,
                       upsampling_mode=upsampling_mode)

  model = Model(inputs=[analysis_tra["inputs"]], outputs=[out])

  return model


def get_net_dualPlane(filterFactor=1, dropout_rate=0, batch_normalization=False, upsampling_mode="transpose_conv",
                        volumeSize_slices=38):
  analysis_tra = analysis_path(2, (volumeSize_slices, 4*volumeSize_slices, 4*volumeSize_slices), batch_normalization,
                               up_strides=(2, 1, 1), pool_size=(1, 2, 2), upsampling_mode=upsampling_mode)
  analysis_sag = analysis_path(2, (4*volumeSize_slices, 4*volumeSize_slices, volumeSize_slices), batch_normalization,
                               up_strides=(1, 1, 2), pool_size=(2, 2, 1), upsampling_mode=upsampling_mode)

  merge = concatenate([analysis_tra["last_pool"], analysis_sag["last_pool"]])
  conv3 = conv3d(merge, int(32 * filterFactor), batch_normalization)
  conv3 = conv3d(conv3, int(64 * filterFactor), batch_normalization)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = conv3d(pool3, int(128 * filterFactor), batch_normalization)
  conv4 = add_dropout(conv4, dropout_rate)
  conv4 = conv3d(conv4, int(128 * filterFactor), batch_normalization)

  out = synthesis_path(conv4,
                       concat3=[conv3],
                       concat2=[analysis_tra["up2"], analysis_sag["up2"]],
                       concat1=[analysis_tra["up1"], analysis_sag["up1"]],
                       filterFactor=filterFactor,
                       dropout_rate=dropout_rate,
                       batch_normalization=batch_normalization,
                       upsampling_mode=upsampling_mode)

  model = Model(
    inputs=[analysis_tra["inputs"], analysis_sag["inputs"]],
    outputs=[out]
  )

  return model


def get_net_triplePlane(filterFactor=1, dropout_rate=0, batch_normalization=False, upsampling_mode="transpose_conv",
                        volumeSize_slices=38):
  analysis_tra = analysis_path(2, (volumeSize_slices, 4*volumeSize_slices, 4*volumeSize_slices), batch_normalization,
                               up_strides=(2, 1, 1), pool_size=(1, 2, 2), upsampling_mode=upsampling_mode, filterFactor= filterFactor)
  analysis_cor = analysis_path(2, (4*volumeSize_slices, volumeSize_slices, 4*volumeSize_slices), batch_normalization,
                               up_strides=(1, 2, 1), pool_size=(2, 1, 2), upsampling_mode=upsampling_mode, filterFactor= filterFactor)
  analysis_sag = analysis_path(2, (4*volumeSize_slices, 4*volumeSize_slices, volumeSize_slices), batch_normalization,
                               up_strides=(1, 1, 2), pool_size=(2, 2, 1), upsampling_mode=upsampling_mode, filterFactor= filterFactor)

  ### merge all paths ###
  merge = concatenate([analysis_tra["last_pool"], analysis_cor["last_pool"], analysis_sag["last_pool"]])
  conv3 = conv3d(merge, int(32 * filterFactor), batch_normalization)
  conv3 = conv3d(conv3, int(64 * filterFactor), batch_normalization)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = conv3d(pool3, int(128 * filterFactor), batch_normalization)
  conv4 = add_dropout(conv4, dropout_rate)
  conv4 = conv3d(conv4, int(128 * filterFactor), batch_normalization)

  out = synthesis_path(conv4,
                       concat3=[conv3],
                       concat2=[analysis_tra["up2"], analysis_sag["up2"], analysis_cor["up2"]],
                       concat1=[analysis_tra["up1"], analysis_sag["up1"], analysis_cor["up1"]],
                       filterFactor=filterFactor,
                       dropout_rate=dropout_rate,
                       batch_normalization=batch_normalization,
                       upsampling_mode=upsampling_mode)

  model = Model(
    inputs=[analysis_tra["inputs"], analysis_cor["inputs"], analysis_sag["inputs"]],
    outputs=[out]
  )

  return model
