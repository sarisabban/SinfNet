'''
MIT License

Copyright (c) 2018 Ayoosh Kathuria / Paperspace <support@paperspace.com>

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


This script is modified from https://github.com/Paperspace/DataAugmentationForObjectDetection
'''

import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def draw_rect(im, cords, color = None):
	'''
	Draw the rectangle on the image
	Parameters
	----------
	im : numpy.ndarray
		numpy image 
	cords: numpy.ndarray
		Numpy array containing bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	Returns
	-------
	numpy.ndarray
		numpy image with bounding boxes drawn on it
	'''
	im = im.copy()
	cords = cords[:,:4]
	cords = cords.reshape(-1,4)
	if not color:
		color = [255,255,255]
	for cord in cords:
		pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
		pt1 = int(pt1[0]), int(pt1[1])
		pt2 = int(pt2[0]), int(pt2[1])
		im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
	return im

def bbox_area(bbox):
	return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
		
def clip_box(bbox, clip_box, alpha):
	'''
	Clip the bounding boxes to the borders of an image
	Parameters
	----------
	bbox: numpy.ndarray
		Numpy array containing bounding boxes of shape `N X 4` where N is the
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	clip_box: numpy.ndarray
		An array of shape (4,) specifying the diagonal co-ordinates of the image
		The coordinates are represented in the format `x1 y1 x2 y2`
	alpha: float
		If the fraction of a bounding box left in the image after being clipped is 
		less than `alpha` the bounding box is dropped.
	Returns
	-------
	numpy.ndarray
		Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes left are being clipped and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	'''
	ar_ = (bbox_area(bbox))
	x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
	y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
	x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
	y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
	bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
	delta_area = ((ar_ - bbox_area(bbox))/ar_)
	mask = (delta_area < (1 - alpha)).astype(int)
	bbox = bbox[mask == 1,:]
	return bbox

def rotate_im(image, angle):
	'''
	Rotate the image.
	Rotate the image such that the rotated image is enclosed inside the tightest
	rectangle. The area not occupied by the pixels of the original image is colored
	black. 
	Parameters
	----------
	image : numpy.ndarray
		numpy image
	angle : float
		angle by which the image is to be rotated
	Returns
	-------
	numpy.ndarray
		Rotated Image
	'''
	# grab the dimensions of the image and then determine the
	# centre
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	# perform the actual rotation and return the image
	image = cv2.warpAffine(image, M, (nW, nH))
	# image = cv2.resize(image, (w,h))
	return image

def get_corners(bboxes):
	'''
	Get corners of bounding boxes
	Parameters
	----------
	bboxes: numpy.ndarray
		Numpy array containing bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	returns
	-------
	numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`	   
	'''
	width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
	height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
	x1 = bboxes[:,0].reshape(-1,1)
	y1 = bboxes[:,1].reshape(-1,1)
	x2 = x1 + width
	y2 = y1 
	x3 = x1
	y3 = y1 + height
	x4 = bboxes[:,2].reshape(-1,1)
	y4 = bboxes[:,3].reshape(-1,1)
	corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
	return corners

def rotate_box(corners,angle,  cx, cy, h, w):
	'''
	Rotate the bounding box.
	Parameters
	----------
	corners : numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
	angle : float
		angle by which the image is to be rotated
	cx : int
		x coordinate of the center of image (about which the box will be rotated)
	cy : int
		y coordinate of the center of image (about which the box will be rotated)
	h : int 
		height of the image
	w : int 
		width of the image
	Returns
	-------
	numpy.ndarray
		Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
	'''
	corners = corners.reshape(-1,2)
	corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cx
	M[1, 2] += (nH / 2) - cy
	# Prepare the vector to be transformed
	calculated = np.dot(M,corners.T).T
	calculated = calculated.reshape(-1,8)
	return calculated

def get_enclosing_box(corners):
	'''
	Get an enclosing box for ratated corners of a bounding box
	Parameters
	----------
	corners : numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
	Returns 
	-------
	numpy.ndarray
		Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	'''
	x_ = corners[:,[0,2,4,6]]
	y_ = corners[:,[1,3,5,7]]
	xmin = np.min(x_,1).reshape(-1,1)
	ymin = np.min(y_,1).reshape(-1,1)
	xmax = np.max(x_,1).reshape(-1,1)
	ymax = np.max(y_,1).reshape(-1,1)
	final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
	return final

def letterbox_image(img, inp_dim):
	'''
	resize image with unchanged aspect ratio using padding
	Parameters
	----------
	img : numpy.ndarray
		Image 
	inp_dim: tuple(int)
		shape of the reszied image
	Returns
	-------
	numpy.ndarray:
		Resized image
	'''
	inp_dim = (inp_dim, inp_dim)
	img_w, img_h = img.shape[1], img.shape[0]
	w, h = inp_dim
	new_w = int(img_w * min(w/img_w, h/img_h))
	new_h = int(img_h * min(w/img_w, h/img_h))
	resized_image = cv2.resize(img, (new_w,new_h))
	canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)
	canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
	return canvas

class RandomHorizontalFlip(object):
	'''
	Randomly horizontally flips the Image with the probability *p*
	Parameters
	----------
	p: float
		The probability with which the image is flipped
	Returns
	-------
	numpy.ndaaray
		Flipped image in the numpy format of shape `HxWxC`

	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self, img, bboxes):
			img_center = np.array(img.shape[:2])[::-1]/2
			img_center = np.hstack((img_center, img_center))
			if random.random() < self.p:
				img = img[:, ::-1, :]
				bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])
				box_w = abs(bboxes[:, 0] - bboxes[:, 2])
				bboxes[:, 0] -= box_w
				bboxes[:, 2] += box_w
			return img, bboxes

class HorizontalFlip(object):
	'''
	Randomly horizontally flips the Image with the probability *p*
	Parameters
	----------
	p: float
		The probability with which the image is flipped
	Returns
	-------
	numpy.ndaaray
		Flipped image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self):
		pass
	def __call__(self, img, bboxes):
		img_center = np.array(img.shape[:2])[::-1]/2
		img_center = np.hstack((img_center, img_center))
		img = img[:, ::-1, :]
		bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])
		box_w = abs(bboxes[:, 0] - bboxes[:, 2])
		bboxes[:, 0] -= box_w
		bboxes[:, 2] += box_w
		return img, bboxes

class RandomVerticalFlip(object):
	'''
	Randomly Vertically flips the Image with the probability *p*
	Parameters
	----------
	p: float
		The probability with which the image is flipped
	Returns
	-------
	numpy.ndaaray
		Flipped image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self, img, bboxes):
			img_center = np.array(img.shape[:2])[::-1]/2
			img_center = np.hstack((img_center, img_center))
			if random.random() < self.p:
				img = img[::-1, :, :]
				bboxes[:, [1, 3]] += 2*(img_center[[1, 3]] - bboxes[:, [1, 3]])
				box_h = abs(bboxes[:, 1] - bboxes[:, 3])
				bboxes[:, 1] -= box_h
				bboxes[:, 3] += box_h
			return img, bboxes

class VerticalFlip(object):
	'''
	Randomly Vertically flips the Image with the probability *p*
	Parameters
	----------
	p: float
		The probability with which the image is flipped
	Returns
	-------
	numpy.ndaaray
		Flipped image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self):
		pass
	def __call__(self, img, bboxes):
		img_center = np.array(img.shape[:2])[::-1]/2
		img_center = np.hstack((img_center, img_center))
		img = img[::-1, :, :]
		bboxes[:, [1, 3]] += 2*(img_center[[1, 3]] - bboxes[:, [1, 3]])
		box_h = abs(bboxes[:, 1] - bboxes[:, 3])
		bboxes[:, 1] -= box_h
		bboxes[:, 3] += box_h
		return img, bboxes

class RandomScale(object):
	'''
	Randomly scales an image	
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	scale: float or tuple(float)
		if **float**, the image is scaled by a factor drawn 
		randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
		the `scale` is drawn randomly from values specified by the 
		tuple
	Returns
	-------
	numpy.ndaaray
		Scaled image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, scale = 0.2, diff = False):
		self.scale = scale
		if type(self.scale) == tuple:
			assert len(self.scale) == 2, "Invalid range"
			assert self.scale[0] > -1, "Scale factor can't be less than -1"
			assert self.scale[1] > -1, "Scale factor can't be less than -1"
		else:
			assert self.scale > 0, "Please input a positive float"
			self.scale = (max(-1, -self.scale), self.scale)
		self.diff = diff
	def __call__(self, img, bboxes):
		#Chose a random digit to scale by 
		img_shape = img.shape
		if self.diff:
			scale_x = random.uniform(*self.scale)
			scale_y = random.uniform(*self.scale)
		else:
			scale_x = random.uniform(*self.scale)
			scale_y = scale_x
		resize_scale_x = 1 + scale_x
		resize_scale_y = 1 + scale_y
		img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
		bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
		canvas = np.zeros(img_shape, dtype = np.uint8)
		y_lim = int(min(resize_scale_y,1)*img_shape[0])
		x_lim = int(min(resize_scale_x,1)*img_shape[1])
		canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
		img = canvas
		bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
		return img, bboxes

class Scale(object):
	'''
	Scales the image	
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	scale_x: float
		The factor by which the image is scaled horizontally
	scale_y: float
		The factor by which the image is scaled vertically
	Returns
	-------
	numpy.ndaaray
		Scaled image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, scale_x = 0.2, scale_y = 0.2):
		self.scale_x = scale_x
		self.scale_y = scale_y
	def __call__(self, img, bboxes):
		#Chose a random digit to scale by 
		img_shape = img.shape
		resize_scale_x = 1 + self.scale_x
		resize_scale_y = 1 + self.scale_y
		img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
		bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
		canvas = np.zeros(img_shape, dtype = np.uint8)
		y_lim = int(min(resize_scale_y,1)*img_shape[0])
		x_lim = int(min(resize_scale_x,1)*img_shape[1])
		canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
		img = canvas
		bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
		return img, bboxes

class RandomTranslate(object):
	'''
	Randomly Translates the image
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	translate: float or tuple(float)
		if **float**, the image is translated by a factor drawn 
		randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
		`translate` is drawn randomly from values specified by the 
		tuple
	Returns
	-------
	numpy.ndaaray
		Translated image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, translate = 0.2, diff = False):
		self.translate = translate
		if type(self.translate) == tuple:
			assert len(self.translate) == 2, "Invalid range"
			assert self.translate[0] > 0 & self.translate[0] < 1
			assert self.translate[1] > 0 & self.translate[1] < 1
		else:
			assert self.translate > 0 and self.translate < 1
			self.translate = (-self.translate, self.translate)
		self.diff = diff
	def __call__(self, img, bboxes):
		#Chose a random digit to scale by 
		img_shape = img.shape
		#translate the image
		#percentage of the dimension of the image to translate
		translate_factor_x = random.uniform(*self.translate)
		translate_factor_y = random.uniform(*self.translate)
		if not self.diff:
			translate_factor_y = translate_factor_x
		canvas = np.zeros(img_shape).astype(np.uint8)
		corner_x = int(translate_factor_x*img.shape[1])
		corner_y = int(translate_factor_y*img.shape[0])
		#change the origin to the top-left corner of the translated box
		orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
		mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
		canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
		img = canvas
		bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
		bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
		return img, bboxes

class Translate(object):
	'''
	Randomly Translates the image 
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	translate: float or tuple(float)
		if **float**, the image is translated by a factor drawn 
		randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
		`translate` is drawn randomly from values specified by the 
		tuple
	Returns
	-------
	numpy.ndaaray
		Translated image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, translate_x = 0.2, translate_y = 0.2, diff = False):
		self.translate_x = translate_x
		self.translate_y = translate_y
		assert self.translate_x > 0 and self.translate_x < 1
		assert self.translate_y > 0 and self.translate_y < 1
	def __call__(self, img, bboxes):
		#Chose a random digit to scale by 
		img_shape = img.shape
		#translate the image
		#percentage of the dimension of the image to translate
		translate_factor_x = self.translate_x
		translate_factor_y = self.translate_y
		canvas = np.zeros(img_shape).astype(np.uint8)
		#get the top-left corner co-ordinates of the shifted box 
		corner_x = int(translate_factor_x*img.shape[1])
		corner_y = int(translate_factor_y*img.shape[0])
		#change the origin to the top-left corner of the translated box
		orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
		mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
		canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
		img = canvas
		bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
		bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
		return img, bboxes
	
class RandomRotate(object):
	'''
	Randomly rotates an image
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	angle: float or tuple(float)
		if **float**, the image is rotated by a factor drawn 
		randomly from a range (-`angle`, `angle`). If **tuple**,
		the `angle` is drawn randomly from values specified by the 
		tuple
	Returns
	-------
	numpy.ndaaray
		Rotated image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, angle = 10):
		self.angle = angle
		if type(self.angle) == tuple:
			assert len(self.angle) == 2, "Invalid range"  
		else:
			self.angle = (-self.angle, self.angle)
	def __call__(self, img, bboxes):
		angle = random.uniform(*self.angle)
		w,h = img.shape[1], img.shape[0]
		cx, cy = w//2, h//2
		img = rotate_im(img, angle)
		corners = get_corners(bboxes)
		corners = np.hstack((corners, bboxes[:,4:]))
		corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
		new_bbox = get_enclosing_box(corners)
		scale_factor_x = img.shape[1] / w
		scale_factor_y = img.shape[0] / h
		img = cv2.resize(img, (w,h))
		new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
		bboxes	= new_bbox
		bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
		return img, bboxes

class Rotate(object):
	'''
	Rotates an image	
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	angle: float
		The angle by which the image is to be rotated 
	Returns
	-------
	numpy.ndaaray
		Rotated image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, angle):
		self.angle = angle
	def __call__(self, img, bboxes):
		'''
		Args:
			img (PIL Image): Image to be flipped.
		Returns:
			PIL Image: Randomly flipped image.
		'''
		angle = self.angle
		print(self.angle)
		w,h = img.shape[1], img.shape[0]
		cx, cy = w//2, h//2
		corners = get_corners(bboxes)
		corners = np.hstack((corners, bboxes[:,4:]))
		img = rotate_im(img, angle)
		corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
		new_bbox = get_enclosing_box(corners)
		scale_factor_x = img.shape[1] / w
		scale_factor_y = img.shape[0] / h
		img = cv2.resize(img, (w,h))
		new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
		bboxes	= new_bbox
		bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
		return img, bboxes

class RandomShear(object):
	'''
	Randomly shears an image in horizontal direction   
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	shear_factor: float or tuple(float)
		if **float**, the image is sheared horizontally by a factor drawn 
		randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
		the `shear_factor` is drawn randomly from values specified by the 
		tuple
	Returns
	-------
	numpy.ndaaray
		Sheared image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, shear_factor = 0.2):
		self.shear_factor = shear_factor
		if type(self.shear_factor) == tuple:
			assert len(self.shear_factor) == 2, "Invalid range for scaling factor"	 
		else:
			self.shear_factor = (-self.shear_factor, self.shear_factor)
		shear_factor = random.uniform(*self.shear_factor)
	def __call__(self, img, bboxes):
		shear_factor = random.uniform(*self.shear_factor)
		w,h = img.shape[1], img.shape[0]
		if shear_factor < 0:
			img, bboxes = HorizontalFlip()(img, bboxes)
		M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
		nW =  img.shape[1] + abs(shear_factor*img.shape[0])
		bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 
		img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
		if shear_factor < 0:
			img, bboxes = HorizontalFlip()(img, bboxes)
		img = cv2.resize(img, (w,h))
		scale_factor_x = nW / w
		bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
		return img, bboxes
		
class Shear(object):
	'''
	Shears an image in horizontal direction   
	Bounding boxes which have an area of less than 25% in the remaining in the 
	transformed image is dropped. The resolution is maintained, and the remaining
	area if any is filled by black color.
	Parameters
	----------
	shear_factor: float
		Factor by which the image is sheared in the x-direction
	Returns
	-------
	numpy.ndaaray
		Sheared image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Tranformed bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, shear_factor = 0.2):
		self.shear_factor = shear_factor
	def __call__(self, img, bboxes):
		shear_factor = self.shear_factor
		if shear_factor < 0:
			img, bboxes = HorizontalFlip()(img, bboxes)
		M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
		nW =  img.shape[1] + abs(shear_factor*img.shape[0])
		bboxes[:,[0,2]] += ((bboxes[:,[1,3]])*abs(shear_factor)).astype(int) 
		img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
		if shear_factor < 0:
			 img, bboxes = HorizontalFlip()(img, bboxes)
		return img, bboxes
	
class Resize(object):
	'''
	Resize the image in accordance to `image_letter_box` function in darknet 
	The aspect ratio is maintained. The longer side is resized to the input 
	size of the network, while the remaining space on the shorter side is filled 
	with black color. **This should be the last transform**
	Parameters
	----------
	inp_dim : tuple(int)
		tuple containing the size to which the image will be resized.
	Returns
	-------
	numpy.ndaaray
		Sheared image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Resized bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, inp_dim):
		self.inp_dim = inp_dim
	def __call__(self, img, bboxes):
		w,h = img.shape[1], img.shape[0]
		img = letterbox_image(img, self.inp_dim)
		scale = min(self.inp_dim/h, self.inp_dim/w)
		bboxes[:,:4] *= (scale)
		new_w = scale*w
		new_h = scale*h
		inp_dim = self.inp_dim	 
		del_h = (inp_dim - new_h)/2
		del_w = (inp_dim - new_w)/2
		add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
		bboxes[:,:4] += add_matrix
		img = img.astype(np.uint8)
		return img, bboxes 

class RandomHSV(object):
	'''
	HSV Transform to vary hue saturation and brightness
	Hue has a range of 0-179
	Saturation and Brightness have a range of 0-255. 
	Chose the amount you want to change thhe above quantities accordingly. 
	Parameters
	----------
	hue : None or int or tuple (int)
		If None, the hue of the image is left unchanged. If int, 
		a random int is uniformly sampled from (-hue, hue) and added to the 
		hue of the image. If tuple, the int is sampled from the range 
		specified by the tuple.   
	saturation : None or int or tuple(int)
		If None, the saturation of the image is left unchanged. If int, 
		a random int is uniformly sampled from (-saturation, saturation) 
		and added to the hue of the image. If tuple, the int is sampled
		from the range specified by the tuple.   
	brightness : None or int or tuple(int)
		If None, the brightness of the image is left unchanged. If int, 
		a random int is uniformly sampled from (-brightness, brightness) 
		and added to the hue of the image. If tuple, the int is sampled
		from the range specified by the tuple.   
	Returns
	-------
	numpy.ndaaray
		Transformed image in the numpy format of shape `HxWxC`
	numpy.ndarray
		Resized bounding box co-ordinates of the format `n x 4` where n is 
		number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
	'''
	def __init__(self, hue = None, saturation = None, brightness = None):
		if hue:
			self.hue = hue 
		else:
			self.hue = 0
		if saturation:
			self.saturation = saturation 
		else:
			self.saturation = 0
		if brightness:
			self.brightness = brightness
		else:
			self.brightness = 0
		if type(self.hue) != tuple:
			self.hue = (-self.hue, self.hue)
		if type(self.saturation) != tuple:
			self.saturation = (-self.saturation, self.saturation)
		if type(brightness) != tuple:
			self.brightness = (-self.brightness, self.brightness)
	def __call__(self, img, bboxes):
		hue = random.randint(*self.hue)
		saturation = random.randint(*self.saturation)
		brightness = random.randint(*self.brightness)
		img = img.astype(int)
		a = np.array([hue, saturation, brightness]).astype(int)
		img += np.reshape(a, (1,1,3))
		img = np.clip(img, 0, 255)
		img[:,:,0] = np.clip(img[:,:,0],0, 179)
		img = img.astype(np.uint8)
		return img, bboxes
	
class Sequence(object):
	'''
	Initialise Sequence object
	Apply a Sequence of transformations to the images/boxes.
	Parameters
	----------
	augemnetations : list 
		List containing Transformation Objects in Sequence they are to be 
		applied
	probs : int or list 
		If **int**, the probability with which each of the transformation will 
		be applied. If **list**, the length must be equal to *augmentations*. 
		Each element of this list is the probability with which each 
		corresponding transformation is applied
	Returns
	-------
	Sequence
		Sequence Object 
	'''
	def __init__(self, augmentations, probs = 1):
		self.augmentations = augmentations
		self.probs = probs
	def __call__(self, images, bboxes):
		for i, augmentation in enumerate(self.augmentations):
			if type(self.probs) == list:
				prob = self.probs[i]
			else:
				prob = self.probs
			if random.random() < prob:
				images, bboxes = augmentation(images, bboxes)
		return images, bboxes
