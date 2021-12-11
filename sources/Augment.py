import os
import cv2
import PIL
import json
import keras
import numpy as np
import imgaug as ia
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict
from imgaug import augmenters as iaa
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from .DAOD import *

def augment_cnn(input_path='./dataset/Train',
				output_path='./dataset/Augmented',
				count=10):
	''' Augments images and saves them into a new directory '''
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
		name  = Images[:-4]
		size  = img.size
		image = img_to_array(img)
		image = image.reshape(1, size[1], size[0], 3)
		image = image.astype('float32')
		gen.fit(image)
		images_flow = gen.flow(image, batch_size=1)
		for i, new_images in enumerate(images_flow):
			new_image = array_to_img(new_images[0], scale=True)
			output = '{}/Aug_{}-{}.jpg'.format(output_path, name, i+1)
			print(output)
			new_image.save(output)
			if i >= count-1: break

def augment_bbox(image_input='./dataset/Train',
				image_output='./dataset/Augmented',
				bbox_input='./dataset/Annotations',
				bbox_output='./dataset/Augmented_Annotations',
				count=10,
				input_format='csv',
				output_format='csv'):
	'''Augment images with bounding boxes and saves them into a new directory'''
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
					BBOX[filename].append([x, y, x+w, y+h, label])
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
				BBOX[filename].append([x, y, x+w, y+h, label])
	elif input_format == 'xml':
		TheLines = []
		BBOX = defaultdict(list)
		for item in os.listdir(bbox_input):
			root = ET.parse('{}/{}'.format(bbox_input, item)).getroot()
			for o in root:
				if o.tag == 'filename':
					filename = o.text
				if o.tag == 'object':
					for b in o:
						if b.tag == 'name':
							label = b.text
						if b.tag == 'bndbox':
							x = float(b[0].text)
							y = float(b[1].text)
							w = float(b[2].text)
							h = float(b[3].text)
							BBOX[filename].append([x, y, x+w, y+h, label])
	for Images in os.listdir(image_input):
		Iname = Images[:-4]
		img = cv2.imread('{}/{}'.format(image_input, Images))[:,:,::-1]
		bboxes = np.array(BBOX[Images], dtype=object)
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
					header1 = 'filename,file_size,file_attributes,'
					header2 = 'region_count,region_id,region_shape_attributes,'
					header2 = 'region_attributes\n'
					header = header1 + header2 + header3
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
						size=os.stat('{}/{}'.format(image_input,Images)).st_size
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
					f.write('\t<path>../dataset/Train/{}.jpg</path>\n'\
					.format(filename))
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
			print('{}'.format(Ioutput))

def augment_poly(image_input='dataset/1.jpg', image_output='/dataset', poly_input='dataset/1.json', poly_output='/dataset', count=2):
	'''
	MIT License

	Copyright (c) 2019 Seth Adams

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

	Adapted from:
	https://github.com/seth814/Semantic-Shapes
	https://www.youtube.com/watch?v=-Z7Sx2sS8z8
	'''
	''' Augment images with polygons and saves them into a new directory '''
	for iters in range(int(count)):
		seq = iaa.Sequential([
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.Multiply((0.7, 1.0)),
			iaa.Affine(
					translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
					rotate=(-90, 90),
					shear=(-16, 16),
					mode=ia.ALL),
			iaa.Sometimes(0.5, iaa.Dropout((0.001, 0.01), per_channel=0.5))
			], random_order=True)
		seq_det = seq.to_deterministic()
		im = cv2.imread(image_input, 1)
		with open(poly_input) as handle: data = json.load(handle)
		shape_dicts = data['shapes']
		points = []
		aug_shape_dicts = []
		i = 0
		for shape in shape_dicts:
			for pairs in shape['points']:
				points.append(ia.Keypoint(x=pairs[0], y=pairs[1]))
			_d = {}
			_d['label'] = shape['label']
			_d['index'] = (i, i+len(shape['points']))
			aug_shape_dicts.append(_d)
			i += len(shape['points'])
		W, H = Image.open(image_input).size
		keypoints = ia.KeypointsOnImage(points, shape=(H,W,3))# Switch if polygons are in the wrong location
		image_aug = seq_det.augment_images([im])[0]
		keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
		for shape in aug_shape_dicts:
			start, end = shape['index']
			aug_points = [[keypoint.x, keypoint.y] for keypoint in keypoints_aug.keypoints[start:end]]
			shape['points'] = aug_points
		NewName = image_input.split('/')[-1][:-4]
		print('{}/Aug_{}-{}.jpg'.format(image_output, NewName, str(iters+1)))
		cv2.imwrite('{}/Aug_{}-{}.jpg'.format(image_output, NewName, str(iters+1)), image_aug)
		with open('{}/Aug_{}-{}.json'.format(poly_output, NewName, str(iters+1)), 'w+') as f:
			version = data['version']
			flags = data['flags']
			lineColor = data['lineColor']
			fillColor = data['fillColor']
			#path = '.{}/Aug_{}'.format(image_output, image_input.split('/')[-1])
			path = '{}/Aug_{}-{}.jpg'.format(image_output, NewName, str(iters+1))
			imageData = data['imageData']
			W, H = Image.open(image_input).size
			# files have the 'imagePath' value incorrect, it should be '../Train/{}'.format('FILENAME.json')
			header = '{{"version": "{}",\n"flags": {},\n"lineColor": {},\n"fillColor": {},\n"imagePath": "{}",\n"imageData": "{}",\n"imageHeight": {},\n"imageWidth": {},\n"shapes": ['\
			.format(version, flags, lineColor, fillColor, path, imageData, W, H)
			f.write(header)
			for info in aug_shape_dicts:
				shape_type = 'polygon'
				line_color = 'null'
				fill_color = 'null'
				label = info['label']
				points = info['points']
				body = '\n\t{{"label": "{}",\n\t\t"line_color": {},\n\t\t"fill_color": {},\n\t\t"points": {},\n\t\t"shape_type": "{}"}},'\
				.format(label, line_color, fill_color, points, shape_type)
				f.write(body)
			loc = f.seek(0, os.SEEK_END)
			f.seek(loc-1)
			f.write(']}')

def rescale_poly(image='1.jpg', annot='2.json', scale=None, H=None, show=False):
	''' Rescale an image and its polygon annotations '''
	img = cv2.imread(image)
	scale_percent = scale
	if H == None:
		W = int(img.shape[1] * scale_percent / 100)
		H = int(img.shape[0] * scale_percent / 100)
		resize = cv2.resize(img, (W, H))
		data = json.load(open(annot))
		for p in data['shapes']:
			points = np.array(p['points'])
			points = points.reshape((-1, 1, 2))
			points = np.array(points * scale_percent / 100, dtype=np.int)
			if show:
				resize = cv2.polylines(resize, [points], True, (255, 69, 0), 2)
			points = points.reshape((-1, 2)).tolist()
			p['points'] = points
	if scale_percent == None:
		scale_percent = (img.shape[0] / H)
		W = int(img.shape[1] / scale_percent)
		resize = cv2.resize(img, (W, H))
		data = json.load(open(annot))
		for p in data['shapes']:
			points = np.array(p['points'])
			points = points.reshape((-1, 1, 2))
			points = np.array(points / scale_percent, dtype=np.int)
			if show:
				resize = cv2.polylines(resize, [points], True, (255, 69, 0), 2)
			points = points.reshape((-1, 2)).tolist()
			p['points'] = points
	name = image[:-4].split('/')[1]
	print(name)
	print('Original shape:', img.shape)
	print('Resized shape: ', resize.shape)
	cv2.imwrite('resized_{}.jpg'.format(name), resize)
	with open('resized_{}.json'.format(name), 'w') as json_file:
		json.dump(data, json_file, indent=4, sort_keys=True)
