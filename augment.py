from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

input_path = 'mouth/head.tif'
output_path = 'X/X{}.jpg'
count = 10

gen = ImageDataGenerator(	featurewise_center=True,
							samplewise_center=True,
							featurewise_std_normalization=False,
							samplewise_std_normalization=False,
							zca_whitening=True,
							zca_epsilon=1e-06,
							rotation_range=10,
							width_shift_range=30,
							height_shift_range=30,
							brightness_range=None,
							shear_range=0.0,
							zoom_range=0.0,
							channel_shift_range=0.0,
							fill_mode='nearest',
							cval=0.0,
							horizontal_flip=True,
							vertical_flip=True,
							rescale=None,
							preprocessing_function=None,
							data_format='channels_last',
							validation_split=0.0,
#							interpolation_order=1,
							dtype='float32')

image = img_to_array(load_img(input_path))
image = image.reshape(1, 1040, 1392, 3)
image = image.astype('float32')
gen.fit(image)
images_flow = gen.flow(image, batch_size=1)
for i, new_images in enumerate(images_flow):
	new_image = array_to_img(new_images[0], scale=True)
	new_image.save(output_path.format(i+1))
	if i >= count-1: break
