#!/usr/bin/env python3

import os
import sys
import keras
import sklearn
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix

def CNN(network='VGG16', choice='predict', weights='weights.h5', prediction='./dataset/Test/image.jpg'):
	''' Train images using one of several CNNs '''
	Train   = './dataset/Train'
	Valid   = './dataset/Valid'
	Tests   = './dataset/Test'
	shape   = (224, 224)
	epochs  = 20
	batches = 16
	classes = []
	for c in os.listdir(Train): classes.append(c)
	if network == 'VGG16' or 'vgg16':
		IDG = keras.preprocessing.image.ImageDataGenerator(
			rescale=1./255,
			preprocessing_function=keras.applications.vgg16.preprocess_input)
		train = IDG.flow_from_directory(Train, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		valid = IDG.flow_from_directory(Valid, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		tests = IDG.flow_from_directory(Tests, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=False)
		input_shape = train.image_shape
		model = VGG16(weights=None, input_shape=input_shape,
			classes=len(classes))
	elif network == 'VGG19' or 'vgg19':
		IDG = keras.preprocessing.image.ImageDataGenerator(
			rescale=1./255,
			preprocessing_function=keras.applications.vgg19.preprocess_input)
		train = IDG.flow_from_directory(Train, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		valid = IDG.flow_from_directory(Valid, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		tests = IDG.flow_from_directory(Tests, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=False)
		input_shape = train.image_shape
		model = VGG16(weights=None, input_shape=input_shape,
			classes=len(classes))        
	elif network == 'ResNet50' or 'resnet50':
		IDG = keras.preprocessing.image.ImageDataGenerator(
			rescale=1./255,
			preprocessing_function=keras.applications.resnet50.preprocess_input)
		train = IDG.flow_from_directory(Train, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		valid = IDG.flow_from_directory(Valid, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		tests = IDG.flow_from_directory(Tests, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=False)
		input_shape = train.image_shape
		model = ResNet50(weights=None, input_shape=input_shape,
			classes=len(classes))
	elif network == 'DenseNet201' or 'densenet201':
		IDG = keras.preprocessing.image.ImageDataGenerator(
			rescale=1./255,
			preprocessing_function=keras.applications.densenet.preprocess_input)
		train = IDG.flow_from_directory(Train, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		valid = IDG.flow_from_directory(Valid, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=True)
		tests = IDG.flow_from_directory(Tests, target_size=shape, color_mode='rgb',
			classes=classes, batch_size=batches, shuffle=False)
		input_shape = train.image_shape
		model = DenseNet201(weights=None, input_shape=input_shape,
			classes=len(classes))
	model.compile(optimizer=keras.optimizers.SGD(
			lr=1e-3,
			decay=1e-6,
			momentum=0.9,
			nesterov=True),
			loss='categorical_crossentropy',
			metrics=['accuracy'])
	Esteps = int(train.samples/train.next()[0].shape[0])
	Vsteps = int(valid.samples/valid.next()[0].shape[0])
	Tsteps = int(tests.samples/tests.next()[0].shape[0])
	if choice == 'train':
		history= model.fit_generator(train,
			steps_per_epoch=Esteps,
			epochs=epochs,
			validation_data=valid,
			validation_steps=Vsteps,
			verbose=1)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.show()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.show()
		evaluation = model.evaluate_generator(tests, Tsteps)
		print('Test Set: Accuracy {} Loss {}'.format(
			round(evaluation[1], 4), round(evaluation[0], 4)))
		Y_pred = model.predict_generator(tests, verbose=1)
		y_pred = np.argmax(Y_pred, axis=1)
		matrix = confusion_matrix(tests.classes, y_pred)
		df_cm  = pd.DataFrame(matrix, index=classes, columns=classes)
		plt.figure(figsize=(10, 7))
		sn.heatmap(df_cm, annot=True)
		print(classification_report(tests.classes,y_pred,target_names=classes))
		model.save_weights('weights.h5')
	elif choice == 'predict':
		model.load_weights(weights)
		img = image.load_img(prediction, target_size=shape)
		im = image.img_to_array(img)
		im = np.expand_dims(im, axis=0)
		if network == 'VGG16' or 'vgg16':
			im = keras.applications.vgg16.preprocess_input(im)
			prediction = model.predict(im)
			print(prediction)
		elif network == 'VGG19' or 'vgg19':
			im = keras.applications.vgg19.preprocess_input(im)
			prediction = model.predict(im)
			print(prediction)
		elif network == 'ResNet50' or 'resnet50':
			im = keras.applications.resnet50.preprocess_input(im)
			prediction = model.predict(im)
			print(prediction)
			print(keras.applications.resnet50.decode_predictions(prediction))
		elif network == 'DenseNet201' or 'densenet201':
			im = keras.applications.densenet201.preprocess_input(im)
			prediction = model.predict(im)
			print(prediction)
			print(keras.applications.densenet201.decode_predictions(prediction))

if __name__ == '__main__':
	CNN(network='VGG16',
		choice='predict',
		weights='weights.h5',
		prediction='./dataset/Test/image.jpg')
