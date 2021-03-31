#!/usr/bin/env python3

import os
import sys
import cv2
import math
import json
import numpy as np
import imgaug as ia
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict

def Biomass(W, H, D, w, h, whitePX):
	'''
	Approximate biomass from semantic segmentation output
	Adapted from https://doi.org/10.1016/j.soilbio.2019.03.021
	'''
	Width_img = W # size of image in micrometers
	Hight_img = H # size of image in micrometers
	Depth_img = D # size of image in micrometers
	width = w # size of image in pixels
	hight = h # size of image in pixels
	pixel = Width_img/width # micrometer value of a pixel
	volume = pixel*whitePX*Depth_img
	print('Volume = {:,} μm^3'.format(round(volume, 3)))
	x = volume*1.7
	biomass = x/1.6e6
	print('Biomass = {:,} μg'.format(biomass))

def crop(filename='test.jpg'):
	''' Crops an image to make its dimetions multiples of 32 '''
	img = Image.open(filename)
	W, H = img.size
	w = 32* math.floor(W/32)
	h = 32* math.floor(H/32)
	area = (0, 0, w, h)
	c_img = img.crop(area)
	c_img.save('{}_cropped.jpg'.format(filename[:-4]))

def segment(filename='test.jpg', size=(1000, 1000)):
	''' Segments a large image into smaller images '''
	name = filename.split('.')[0]
	kernel = size
	stride = kernel
	img = Image.open(filename)
	pix = img.getdata()
	W, H = Image.open(filename).size
	L, U, R, D = 0, 0, kernel[0], kernel[1]
	count = 1
	for hight in range(int(H/kernel[1])):
		for width in range(int(W/kernel[0])):
			box = (L, U, R, D)
			region = img.crop(box)
			region.save('{}_segment_{}.jpg'.format(name, count), 'JPEG')
			count += 1
			L += stride[0]
			R += stride[0]
		L = 0
		U += stride[1]
		R = stride[0]
		D += stride[1]

def confirm_box(image_path='dataset/1.jpg', annotation_path='dataset/1.xml'):
	''' Plot augmented image and box to confirm correct box augmentation '''
	img = cv2.imread(image_path)[:,:,::-1].copy()
	root = ET.parse(annotation_path).getroot()
	BBOX = defaultdict(list)
	for o in root:
		if o.tag == 'filename':
			filename = o.text
		if o.tag == 'object':
			for b in o:
				if b.tag == 'name':
					label = 0#b.text
				if b.tag == 'bndbox':
					x = float(b[0].text)
					y = float(b[1].text)
					w = float(b[2].text)
					h = float(b[3].text)
					BBOX[filename].append([x, y, w, h, label])
	bboxes = np.array(BBOX[filename])
	cords = bboxes[:,:4]
	cords = cords.reshape(-1,4)
	for cord in cords:
		pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
		pt1 = int(pt1[0]), int(pt1[1])
		pt2 = int(pt2[0]), int(pt2[1])
		im = cv2.rectangle(img, pt1, pt2, (255, 69, 0), 10)
	plt.imshow(im)
	plt.show()

def confirm_poly(image_path='dataset/1.jpg',annotation_path='dataset/1.json'):
	''' Plot augmented image and polygon to confirm polygon augmentation '''
	img = cv2.imread(image_path, 1)
	with open(annotation_path) as handle: data = json.load(handle)
	for shapes in data['shapes']:
		points = np.array(shapes['points'], np.int32)
		points = points.reshape((-1, 1, 2)) 
		image = cv2.polylines(img, [points], True, (255, 69, 0), 10)
	plt.imshow(image)
	plt.show()
