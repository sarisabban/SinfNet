import os
import sys
import cv2
import json
import random
import numpy as np
import tensorflow as tf
#import pydensecrf.densecrf as dcrf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
#from pydensecrf.utils import unary_from_softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Lambda, Conv2DTranspose, Add
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate

imshape = (256, 256, 3)
mode = sys.argv[2]			# classification mode (binary or multi)
model_name = 'unet_'+mode	# model_name (unet or fcn_8)
LABELS = sys.argv[3:]
hues = {}
for l in LABELS:
	hues[l] = random.randint(0, 360)
labels = sorted(hues.keys())
if mode == 'binary': n_classes = 1
elif mode == 'multi': n_classes = len(labels) + 1
assert imshape[0]%32 == 0 and imshape[1]%32 == 0,\
    "imshape should be multiples of 32. comment out to test different imshapes."

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, annot_paths, batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]
        X, y = self.__data_generation(image_paths, annot_paths)
        return X, y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def get_poly(self, annot_path):
        with open(annot_path) as handle:
            data = json.load(handle)
        shape_dicts = data['shapes']
        return shape_dicts
    def create_binary_masks(self, im, shape_dicts):
        blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        for shape in shape_dicts:
            if shape['label'] != 'background':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(blank, [points], 255)
        blank = blank / 255.0
        return np.expand_dims(blank, axis=2)
    def create_multi_masks(self, im, shape_dicts):
        channels = []
        cls = [x['label'] for x in shape_dicts]
        poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
        label2poly = dict(zip(cls, poly))
        background = np.zeros(shape=(im.shape[0], im.shape[1]),dtype=np.float32)
        for i, label in enumerate(labels):
            blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
            if label in cls:
                cv2.fillPoly(blank, [label2poly[label]], 255)
                cv2.fillPoly(background, [label2poly[label]], 255)
            channels.append(blank)
        if 'background' in cls:
            background = np.zeros(shape=(im.shape[0],
                                         im.shape[1]), dtype=np.float32)
            cv2.fillPoly(background, [label2poly['background']], 255)
        else:
            _, background = cv2.threshold(background, 127, 255,
                                          cv2.THRESH_BINARY_INV)
        channels.append(background)
        Y = np.stack(channels, axis=2) / 255.0
        return Y
    def __data_generation(self, image_paths, annot_paths):
        X = np.empty((self.batch_size,
                      imshape[0], imshape[1], imshape[2]), dtype=np.float32)
        Y = np.empty((self.batch_size,
                      imshape[0], imshape[1], n_classes),  dtype=np.float32)
        for i, (im_path, annot_path) in enumerate(zip(image_paths,annot_paths)):
            if imshape[2] == 1:
                im = cv2.imread(im_path, 0)
                im = np.expand_dims(im, axis=2)
            elif imshape[2] == 3:
                im = cv2.imread(im_path, 1)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            shape_dicts = self.get_poly(annot_path)
            if n_classes == 1:
                mask = self.create_binary_masks(im, shape_dicts)
            elif n_classes > 1:
                mask = self.create_multi_masks(im, shape_dicts)
            X[i,] = im
            Y[i,] = mask
        return X, Y

def unet(pretrained=False, base=4):
    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            model.summary()
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))
    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'
    b = base
    i = Input((imshape[0], imshape[1], imshape[2]))
    s = Lambda(lambda x: preprocess_input(x)) (i)
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(2**(b+3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(2**(b+1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    u9 = Conv2DTranspose(2**b, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(2**b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    o = Conv2D(n_classes, (1, 1), activation=final_act) (c9)
    model = Model(inputs=i, outputs=o, name=model_name)
    model.compile(optimizer=Adam(1e-4), loss=loss, metrics=[dice])
    #model.summary()
    return model

def fcn_8(pretrained=False, base=4):
    if pretrained:
        path = os.path.join('models', model_name+'.model')
        if os.path.exists(path):
            model = load_model(path, custom_objects={'dice': dice})
            return model
        else:
            print('Failed to load existing model at: {}'.format(path))
    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'
    b = base
    i = Input(shape=imshape)
    s = Lambda(lambda x: preprocess_input(x)) (i)
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv1')(s)
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv1')(x)
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv1')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv2')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = x
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv1')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    conv6 = Conv2D(2048 , (7, 7) , activation='elu' , padding='same', name="conv6")(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048 , (1, 1) , activation='elu' , padding='same', name="conv7")(conv6)
    conv7 = Dropout(0.5)(conv7)
    pool4_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    u2_skip = Add()([pool4_n, u2])
    pool3_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    u4_skip = Add()([pool3_n, u4])
    o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same', activation=final_act)(u4_skip)
    model = Model(inputs=i, outputs=o, name=model_name)
    model.compile(optimizer=Adam(1e-4), loss=loss, metrics=[dice])
    #model.summary()
    return model

def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f) + smooth)

def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)
    for i, label in enumerate(labels):
        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,i].astype(np.uint8)
        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)
    return blank

def crf(im_softmax, im_rgb):
    n_classes = im_softmax.shape[2]
    feat_first = im_softmax.transpose((2, 0, 1)).reshape(n_classes, -1)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    im_rgb = np.ascontiguousarray(im_rgb)
    d = dcrf.DenseCRF2D(im_rgb.shape[1], im_rgb.shape[0], n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=im_rgb,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((im_rgb.shape[0], im_rgb.shape[1]))
    if mode is 'binary':
        return res * 255.0
    if mode is 'multi':
        res_hot = to_categorical(res) * 255.0
        res_crf = add_masks(res_hot)
        return res_crf

def semantic_train():
	image_paths=[os.path.join('./dataset/Train', x) for x in sorted_fns('./dataset/Train')]
	annot_paths=[os.path.join('./dataset/Train_Annotations', x) for x in sorted_fns('./dataset/Train_Annotations')]
	if 'unet' in model_name:
		model = unet(pretrained=False, base=4)
	elif 'fcn_8' in model_name:
		model = fcn_8(pretrained=False, base=4)
	tg = DataGenerator(image_paths=image_paths,
                    annot_paths=annot_paths,
                    batch_size=5)
	checkpoint = ModelCheckpoint(os.path.join('models', model_name+'.model'),
	monitor='dice', verbose=1, mode='max', save_best_only=True,
    save_weights_only=False, period=10)
	model.fit_generator(generator=tg,
                     steps_per_epoch=len(tg),
                     epochs=500,
                     verbose=1,
                     callbacks=[checkpoint])

def semantic_predict(filename, CALC_CRF=True):
    model = load_model(os.path.join('models', model_name+'.model'),
                       custom_objects={'dice': dice})
    im_cv = cv2.imread(filename)
    im = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB).copy()
    tmp = np.expand_dims(im, axis=0)
    roi_pred = model.predict(tmp)
    if n_classes == 1:
        roi_mask = roi_pred.squeeze()*255.0
        roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
    elif n_classes > 1:
        roi_mask = add_masks(roi_pred.squeeze()*255.0)
    if CALC_CRF:
        if n_classes == 1:
            roi_pred = roi_pred.squeeze()
            roi_softmax = np.stack([1-roi_pred, roi_pred], axis=2)
            roi_mask = crf(roi_softmax, im)
            roi_mask = np.array(roi_mask, dtype=np.float32)
            roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
        elif n_classes > 1:
            roi_mask = crf(roi_pred.squeeze(), im)
    cv2.imwrite('output.jpg', roi_mask)

if __name__ == '__main__':
	semantic_train()
	semantic_predict('0.jpg')
