from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

input_path = 'mouth/head.tif'
output_path = 'X/X{}.jpg'
count = 3

gen = ImageDataGenerator(rotation_range=20,
						width_shift_range=0.2,
						height_shift_range=0.2,
						horizontal_flip=True)
image = img_to_array(load_img(input_path))
image = image.reshape((1,) + image.shape)
images_flow = gen.flow(image, batch_size=1)
for i, new_images in enumerate(images_flow):
	new_image = array_to_img(new_images[0], scale=True)
	new_image.save(output_path.format(i+1))
	if i >= count-1: break
