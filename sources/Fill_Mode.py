import cv2
import random
import numpy as np
from scipy import ndimage
from matplotlib import image
import matplotlib.pyplot as plt
from collections import defaultdict

def rotate_im(image, angle):
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	image = cv2.warpAffine(image, M, (nW, nH))
	return image

def draw_rect(im, cords, color = None):
	im = im.copy()
	cords = cords[:,:4]
	cords = cords.reshape(-1,4)
	if not color:
		color = [255,255,255]
	for cord in cords:
		pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
		pt1 = int(pt1[0]), int(pt1[1])
		pt2 = int(pt2[0]), int(pt2[1])
		im = cv2.rectangle(im.copy(), pt1, pt2, color,\
		int(max(im.shape[:2])/200))
	return im

class RandomHSV(object):
	def __init__(self, hue = None, saturation = None, brightness = None):
		if hue: self.hue = hue
		else: self.hue = 0
		if saturation: self.saturation = saturation
		else: self.saturation = 0
		if brightness: self.brightness = brightness
		else: self.brightness = 0
		if type(self.hue) != tuple: self.hue = (-self.hue, self.hue)
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

def clip_box(bbox, clip_box, alpha):
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

def bbox_area(bbox): return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def get_enclosing_box(corners):
	x_ = corners[:,[0,2,4,6]]
	y_ = corners[:,[1,3,5,7]]
	xmin = np.min(x_,1).reshape(-1,1)
	ymin = np.min(y_,1).reshape(-1,1)
	xmax = np.max(x_,1).reshape(-1,1)
	ymax = np.max(y_,1).reshape(-1,1)
	final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
	return final

def rotate_box(corners,angle,  cx, cy, h, w):
	corners = corners.reshape(-1,2)
	corners = np.hstack((corners, np.ones((corners.shape[0],1),\
	dtype = type(corners[0][0]))))
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	M[0, 2] += (nW / 2) - cx
	M[1, 2] += (nH / 2) - cy
	calculated = np.dot(M,corners.T).T
	calculated = calculated.reshape(-1,8)
	return calculated

def get_corners(bboxes):
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

class Rotate(object):
	def __init__(self, angle):
		self.angle = angle
	def __call__(self, img, bboxes):
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
		new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x,\
		scale_factor_y]
		bboxes = new_bbox
		bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
		return img, bboxes

def transform_matrix_offset_center(matrix, x, y):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix

#### COULD NOT FIGURE OUT IMPLEMENTING FILL_MODE AND EXPORT UPDATED BBOXES ####

def rotate(img, bboxes, angle=0):
	theta = np.deg2rad(-angle)
	transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
								[np.sin(theta), np.cos(theta), 0],
								[0, 0, 1]])
#		 translate_matrix = np.array([[1, 0, tx],
#								[0, 1, ty],
#								[0, 0, 1]])
#		 shear_matrix = np.array([[1, -np.sin(shear), 0],
#								[0, np.cos(shear), 0],
#								[0, 0, 1]])
#		 scale_matrix = np.array([[zx, 0, 0],
#								[0, zy, 0],
#								[0, 0, 1]])

	w,h = img.shape[1], img.shape[0]
	transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
	final_affine_matrix = transform_matrix[:2, :2]
	final_offset = transform_matrix[:2, 2]
	img = np.rollaxis(img, 2, 0)
	channel_images = [ndimage.interpolation.affine_transform(
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
	img = rotate_im(img, angle)
	corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
	new_bbox = get_enclosing_box(corners)
	scale_factor_x = img.shape[1] / w
	scale_factor_y = img.shape[0] / h
	image = cv2.resize(img, (w,h))
	new_bbox[:,:4] /= [	scale_factor_x,
						scale_factor_y,
						scale_factor_x,
						scale_factor_y]
	bboxes = new_bbox
	bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
	return img, bboxes









BBOX = defaultdict(list)
with open('messi.txt', 'r') as f:
	filename = 'messi.jpg'
	next(f)
	for line in f:
		line = line.split()
		label = line[4]
		x = int(line[0])
		y = int(line[1])
		w = int(line[2])
		h = int(line[3])
		BBOX[filename].append([x, y, w, h, label])
bboxes = np.array(BBOX[filename], dtype=object)
img = image.imread('messi.jpg')

I = rotate(img, bboxes, 30)
img_, bboxes_ = RandomHSV(20, 20, 20)(I[0].copy(), I[1].copy())

output = draw_rect(img_, bboxes_)
plt.imshow(output)
plt.show()
