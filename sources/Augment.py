import os
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image

def augment(input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=10):
	''' Augments images and saves them into a new directory '''
	os.makedirs('./dataset/Augmented', exist_ok=True)
	for Image in os.listdir(input_path):
		gen = ImageDataGenerator(	featurewise_center=False,
									samplewise_center=False,
									featurewise_std_normalization=False,
									samplewise_std_normalization=False,
									zca_whitening=False,
									zca_epsilon=1e-06,
									rotation_range=30,
									width_shift_range=30,
									height_shift_range=30,
									brightness_range=[0.6, 0.8],
									shear_range=4.0,
									zoom_range=[0.8, 1.0],
									channel_shift_range=10,
									fill_mode='reflect',
									cval=0.0,
									horizontal_flip=True,
									vertical_flip=True,
									rescale=1./255,
									preprocessing_function=None,
									data_format='channels_last',
									validation_split=0.2,
									#interpolation_order=1,
									dtype='float32')
		img   = load_img('{}/{}'.format(input_path, Image))
		name  = Image.split('.')[0]
		size  = img.size
		image = img_to_array(img)
		image = image.reshape(1, size[0], size[1], 3)
		image = image.astype('float32')
		gen.fit(image)
		images_flow = gen.flow(image, batch_size=1)
		for i, new_images in enumerate(images_flow):
			new_image = array_to_img(new_images[0], scale=True)
			output = '{}/Aug_{}-{}.jpg'.format(output_path, name, i+1)
			print(output)
			new_image.save(output)
			if i >= count-1: break

if __name__ == '__main__':
	augment(input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=10)
