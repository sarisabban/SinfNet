#!/usr/bin/env python3

import keras
import sklearn
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet201
from sklearn.metrics import classification_report, confusion_matrix

# Import images and augment
Train   = 'CatDog/train'
Tests   = 'CatDog/tests'
Valid   = 'CatDog/valid'
classes = ['cats', 'dogs']
shape   = (224, 224)
batches = 16
epochs  = 10
IDG = keras.preprocessing.image.ImageDataGenerator()#horizontal_flip=True, vertical_flip=True, rotation_range=45)
train = IDG.flow_from_directory(Train, target_size=shape, color_mode='rgb',
                                classes=classes, batch_size=batches)
tests = IDG.flow_from_directory(Tests, target_size=shape, color_mode='rgb',
                                classes=classes, batch_size=batches)
valid = IDG.flow_from_directory(Valid, target_size=shape, color_mode='rgb',
                                classes=classes, batch_size=batches)
input_shape = train.image_shape

# Setup neural network
model = VGG16(weights=None, input_shape=input_shape, classes=len(classes))
#model = VGG19(weights=None, input_shape=input_shape, classes=len(classes))
#model = ResNet50(weights=None, input_shape=input_shape, classes=len(classes))
#model = DenseNet201(weights=None, input_shape=input_shape, classes=len(classes))
model.compile(optimizer=keras.optimizers.SGD(lr=1e-3,
                                             decay=1e-6,
                                             momentum=0.9,
                                             nesterov=True),
                                             loss='categorical_crossentropy',
                                             metrics=['accuracy'])

# Train model
Esteps = int(train.samples/train.next()[0].shape[0])
Vsteps = int(valid.samples/valid.next()[0].shape[0])
history= model.fit_generator(train,
                             steps_per_epoch=Esteps,
                             epochs=epochs,
                             validation_data=valid,
                             validation_steps=Vsteps,
                             verbose=1)

# Plot loss and accuracy
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

# Plot confusion matrix
Y_pred = model.predict_generator(tests, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
matrix = confusion_matrix(tests.classes, y_pred)
df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
plt.figure(figsize=(10,7))
sn.heatmap(df_cm, annot=True)
print(classification_report(tests.classes, y_pred, target_names=classes))

# Save weights
model.save_weights('weights.h5')

# Prediction
model.load_weights('weights.h5')
prdct = model.predict('1.jpg')
print(prdct)
