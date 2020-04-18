import os
import cv2
import PIL
import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from DAOD import *

print('=======')
def augment(input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=10):
	''' Augments images and saves them into a new directory '''
	os.makedirs('./dataset/Augmented', exist_ok=True)
	for Images in os.listdir(input_path):
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
		img   = load_img('{}/{}'.format(input_path, Images))
		name  = Images.split('.')[0]
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

def DAOD(	image_input='./dataset/Train',
			image_output='./dataset/Augmented',
			bbox_input='./dataset/Annotations',
			bbox_output='./dataset/Augmented_Annotations',
			count=10,
			input_format='csv',
			output_format='xml'):
	''' Data Augmentation For Object Detection - no reflection '''
	if input_format == 'txt':
		BBOX = defaultdict(list)
		for filename in os.listdir(bbox_input):
			with open('{}/{}'.format(bbox_input, filename), 'r') as f:
				filename = filename.split('.')
				filename[-1] = '.jpg'
				filename = ''.join(filename)
				next(f)
				for line in f:
					line = line.split()
					label = line[4]
					x = int(line[0])
					y = int(line[1])
					w = int(line[2])
					h = int(line[3])
					BBOX[filename].append([x, y, w, h, label])
	elif input_format == 'csv':
		TheLines = []
		BBOX = defaultdict(list)
		with open(bbox_input, 'r') as F:
			next(F)
			for line in F:
				line = line.strip().split(':')
				filename = line[0].split(',')[0]
				label = line[5].split('"')[4]
				x = int(line[2].split(',')[0])
				y = int(line[3].split(',')[0])
				w = int(line[4].split(',')[0])
				h = int(line[5].split(',')[0].split('}')[0])
				BBOX[filename].append([x, y, w, h, label])
	elif input_format == 'xml':
		TheLines = []
		BBOX = defaultdict(list)
		for item in os.listdir(bbox_input):
			with open('{}/{}'.format(bbox_input, item), 'r') as F:
				next(F)
				filename = F.readline().strip().split()[0].split('>')[1].split('<')[0]
				Xx = []
				Yy = []
				Ww = []
				Hh = []
				Llabel = []
				for line in F:
					line = line.strip().split()[0]
					if '<xmin>' in line:
						Xx.append(line.split('>')[1].split('<')[0])
					elif '<ymin>' in line:
						Yy.append(line.split('>')[1].split('<')[0])
					elif '<xmax>' in line:
						Ww.append(line.split('>')[1].split('<')[0])
					elif '<ymax>' in line:
						Hh.append(line.split('>')[1].split('<')[0])
					elif '<name>' in line:
						Llabel.append(line.split('>')[1].split('<')[0])
				for x, y, w, h, label in zip(Xx, Yy, Ww, Hh, Llabel):
					x = int(x)
					y = int(y)
					w = int(w)
					h = int(h)
					BBOX[filename].append([x, y, w, h, label])
	for Images in os.listdir(image_input):
		Iname = Images.split('.')[0]
		bboxes = np.array(BBOX[Images], dtype=object)
		img = cv2.imread('{}/{}'.format(image_input, Images))[:,:,::-1]
		for i in range(count):
			seq = Sequence([
				RandomHorizontalFlip(0.5),
				RandomVerticalFlip(0.5),
				RandomRotate(15),
				RandomScale(0.01),
				RandomTranslate(0.1),
				RandomShear(0.1),
				RandomHSV(20, 20, 20)])
			img_, bboxes_ = seq(img.copy(), bboxes.copy())
			Ioutput = '{}/Aug_{}-{}.jpg'.format(image_output, Iname, i+1)
			new_image = array_to_img(img_, scale=True)
			new_image.save(Ioutput)
			list_of_boxes = np.ndarray.tolist(bboxes_)
			if output_format == 'txt':
				Boutput = '{}/Aug_{}-{}.txt'.format(bbox_output, Iname, i+1)
				with open(Boutput, 'w') as f:
					f.write(str(len(list_of_boxes))+'\n')
					for i in list_of_boxes:
						line = ' '.join(str(v) for v in i)+'\n'
						f.write(line)
			elif output_format == 'csv':
				Boutput = '{}/Augmented.csv'.format(bbox_output)
				with open(Boutput, 'a+') as f:
					header = 'filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n'
					line = f.seek(0)
					if f.readline() != header:
						f.write(header)
					items = 0
					for line in bboxes_:
						line = np.ndarray.tolist(line)
						line = [str(i) for i in line]
						x = line[0]
						y = line[1]
						w = line[2]
						h = line[3]
						label = line[4]
						filename = 'Aug_{}-{}.jpg'.format(Iname, i+1)
						total = bboxes_.shape[0]
						size=os.stat('{}/{}'.format(image_input, Images)).st_size
						items += 1
						TheLine = '{},{},"{{}}",{},{},"{{""name"":""rect"",""x"":{},""y"":{},""width"":{},""height"":{}}}","{{""{}"":""""}}"\n'\
						.format(filename, size, total, items, x, y, w, h, label)
						f.write(TheLine)
			elif output_format == 'xml':
				Boutput = '{}/Aug_{}-{}.xml'.format(bbox_output, Iname, i+1)
				with open(Boutput, 'w') as f:
					source = 'https://github.com/sarisabban/SinfNet'
					total = bboxes_.shape[0]
					filename = 'Aug_{}-{}'.format(Iname, i+1)
					W, H = Image.open('{}/{}'.format(image_input, Images)).size
					f.write('<annotation>\n')
					f.write('\t<filename>{}.jpg</filename>\n'.format(filename))
					f.write('\t<source>{}</source>\n'.format(source))
					f.write('\t<path>../dataset/Train/{}.jpg</path>\n'.format(filename))
					f.write('\t<size>\n')
					f.write('\t\t<width>{}</width>\n'.format(W))
					f.write('\t\t<height>{}</height>\n'.format(H))
					f.write('\t\t<depth>3</depth>\n')
					f.write('\t</size>\n')
					f.write('\t<segments>{}</segments>\n'.format(total))
					items = 0
					for line in bboxes_:
						line = np.ndarray.tolist(line)
						line = [str(i) for i in line]
						x = line[0]
						y = line[1]
						w = line[2]
						h = line[3]
						label = line[4]
						items += 1
						f.write('\t<object>\n')
						f.write('\t\t<name>{}</name>\n'.format(label))
						f.write('\t\t<bndbox>\n')
						f.write('\t\t\t<xmin>{}</xmin>\n'.format(x))
						f.write('\t\t\t<ymin>{}</ymin>\n'.format(y))
						f.write('\t\t\t<xmax>{}</xmax>\n'.format(w))
						f.write('\t\t\t<ymax>{}</ymax>\n'.format(h))
						f.write('\t\t</bndbox>\n')
						f.write('\t</object>\n')
					f.write('</annotation>')
			#plotted_img = draw_rect(img_, bboxes_)
			#plt.imshow(plotted_img)
			#plt.show()
			print('Completed: {} {}'.format(Ioutput, Boutput))

if __name__ == '__main__':
#	augment(input_path='./dataset/Train',
#			output_path='./dataset/Augmented',
#			count=10)
	DAOD(image_input='./I',
		image_output='./AI',
		bbox_input='./A',
		bbox_output='./AA',
		count=2,
		input_format='txt',
		output_format='xml')








"""
############### COULD NOT FIGURE OUT IMPLEMENTING FILL_MODE ############
import scipy
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def rotate(img, bboxes, angle=0):
    theta = np.deg2rad(-angle)
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    w,h = img.shape[1], img.shape[0]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    img = np.rollaxis(img, 2, 0)
    channel_images = [scipy.ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode='nearest',
        cval=0) for x_channel in img]
    img = np.stack(channel_images, axis=0)
    img = np.rollaxis(img, 0, 2 + 1)

    cx, cy = w//2, h//2
    corners = get_corners(bboxes)
    corners = np.hstack((corners, bboxes[:,4:]))
#    img = rotate_im(img, angle)
    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)
    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h
#    image = cv2.resize(img, (w,h))
    new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    bboxes  = new_bbox
    bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
    return img, bboxes



#        theta = np.deg2rad(theta)
#        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                    [np.sin(theta), np.cos(theta), 0],
#                                    [0, 0, 1]])
#
#        translate_matrix = np.array([[1, 0, tx],
#                                 [0, 1, ty],
#                                 [0, 0, 1]])
#
#        shear_matrix = np.array([[1, -np.sin(shear), 0],
#                                 [0, np.cos(shear), 0],
#                                 [0, 0, 1]])
#
#        scale_matrix = np.array([[zx, 0, 0],
#                                [0, zy, 0],
#                                [0, 0, 1]])







import pickle as pkl
bboxes = pkl.load(open("messi_ann.pkl", "rb"))
img = img_to_array(load_img('messi.jpg'))

I = rotate(img, bboxes, 30)
img_, bboxes_ = RandomHSV(20, 20, 20)(I[0].copy(), I[1].copy())

output = draw_rect(img_, bboxes_)
plt.imshow(output)
plt.show()
"""
