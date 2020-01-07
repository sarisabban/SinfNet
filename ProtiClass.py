#!/usr/bin/env python3

import os
import sys
import glob
import keras
import random
import sklearn
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix

# colors for the bboxes
COLORS = ['red', 'blue','pink', 'cyan', 'green', 'black']

# image sizes for the examples
SIZE = 256, 256

parser = argparse.ArgumentParser(description='Object Detection Neural Network Dataset Labeling')
parser.add_argument('-b', '--bbox',        action='store_true', help='Open the BBox image labeling tool')
parser.add_argument('-t', '--translate',   action='store_true', help='Translate .txt file to .xml file')
parser.add_argument('-k', '--check',       action='store_true', help='Check the images for correct annotation')
parser.add_argument('-e', '--eval',        action='store_true', help='Evaluate the accuracy of the trained neural network')
parser.add_argument('-a', '--augment',     action='store_true', help='Augments images')
parser.add_argument('-cp', '--cnn_predict',action='store_true', help='Predict from CNN')
parser.add_argument('-r', '--rename',      nargs='+', help='Rename a label')
parser.add_argument('-ct','--cnn_train',   nargs='+', help='Train on CNN')
args = parser.parse_args()

class LabelTool():
	''' GUI bounding box annotation tool '''
	'''
	MIT License
	
	Copyright (c) 2017 Shi Qiu
	
	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to
	deal in the Software without restriction, including without limitation the
	rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
	sell copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:
	
	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.
	
	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
	IN THE SOFTWARE.
	
	This script is modified from https://github.com/xiaqunfeng/BBox-Label-Tool
	which is in turn adopted from https://github.com/puzzledqs/BBox-Label-Tool
	'''
	def __init__(self, master, LABELS):
		self.parent = master
		self.parent.title('BBox Label Tool')
		self.frame = Frame(self.parent)
		self.frame.pack(fill=BOTH, expand=1)
		self.parent.resizable(width = FALSE, height = FALSE)
		self.imageDir = ''
		self.imageList= []
		self.egDir = ''
		self.egList = []
		self.outDir = ''
		self.cur = 0
		self.total = 0
		self.category = 0
		self.imagename = ''
		self.labelfilename = ''
		self.tkimg = None
		self.currentLabelclass = ''
		self.cla_can_temp = LABELS
		self.STATE = {}
		self.STATE['click'] = 0
		self.STATE['x'], self.STATE['y'] = 0, 0
		self.bboxIdList = []
		self.bboxId = None
		self.bboxList = []
		self.hl = None
		self.vl = None
		self.srcDirBtn = Button(self.frame, text='Image input folder',
								command=self.selectSrcDir)
		self.srcDirBtn.grid(row=0, column=0)
		self.svSourcePath = StringVar()
		self.entrySrc = Entry(self.frame, textvariable=self.svSourcePath)
		self.entrySrc.grid(row=0, column=1, sticky=W+E)
		self.svSourcePath.set(os.path.join(os.getcwd(),'./dataset/Train'))
		self.ldBtn = Button(self.frame, text="Load Dir", command=self.loadDir)
		self.ldBtn.grid(row=0, column=2, rowspan=2,
						columnspan=2, padx=2, pady=2, ipadx=5, ipady=5)
		self.desDirBtn = Button(self.frame, text='Label output folder',
								command=self.selectDesDir)
		self.desDirBtn.grid(row=1, column=0)
		self.svDestinationPath = StringVar()
		self.entryDes = Entry(self.frame, textvariable=self.svDestinationPath)
		self.entryDes.grid(row=1, column=1, sticky=W+E)
		self.svDestinationPath.set(os.path.join(os.getcwd(),
									'./dataset/BBox_Annotations'))
		self.mainPanel = Canvas(self.frame, cursor='tcross')
		self.mainPanel.bind('<Button-1>', self.mouseClick)
		self.mainPanel.bind('<Motion>', self.mouseMove)
		self.parent.bind('<Escape>', self.cancelBBox)
		self.parent.bind('s', self.cancelBBox)
		self.parent.bind('p', self.prevImage)
		self.parent.bind('n', self.nextImage)
		self.mainPanel.grid(row=2, column=1, rowspan=4, sticky=W+N)
		self.classname = StringVar()
		self.classcandidate = ttk.Combobox(self.frame, state='readonly',
								textvariable=self.classname)
		self.classcandidate.grid(row=2, column=2)
		self.classcandidate['values'] = self.cla_can_temp
		self.classcandidate.current(0)
		self.currentLabelclass = self.classcandidate.get()
		self.btnclass = Button(self.frame, text='Confirm Class',
								command=self.setClass)
		self.btnclass.grid(row=2, column=3, sticky=W+E)
		self.lb1 = Label(self.frame, text='Bounding boxes:')
		self.lb1.grid(row=3, column=2, sticky=W+N)
		self.listbox = Listbox(self.frame, width=22, height=12)
		self.listbox.grid(row=4, column=2, sticky=N+S)
		self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
		self.btnDel.grid(row=4, column=3, sticky=W+E+N)
		self.btnClear = Button(self.frame, text='Clear All',
								command=self.clearBBox)
		self.btnClear.grid(row=4, column=3, sticky=W+E+S)
		self.ctrPanel = Frame(self.frame)
		self.ctrPanel.grid(row=6, column=1, columnspan=2, sticky=W+E)
		self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10,
								command=self.prevImage)
		self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
		self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10,
								command=self.nextImage)
		self.nextBtn.pack(side=LEFT, padx=5, pady=3)
		self.progLabel = Label(self.ctrPanel, text='Progress:     /    ')
		self.progLabel.pack(side=LEFT, padx=5)
		self.tmpLabel = Label(self.ctrPanel, text='Go to Image No.')
		self.tmpLabel.pack(side=LEFT, padx=5)
		self.idxEntry = Entry(self.ctrPanel, width=5)
		self.idxEntry.pack(side=LEFT)
		self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
		self.goBtn.pack(side=LEFT)
		self.disp = Label(self.ctrPanel, text='')
		self.disp.pack(side=RIGHT)
		self.frame.columnconfigure(1, weight=1)
		self.frame.rowconfigure(4, weight=1)
	def selectSrcDir(self):
		path = filedialog.askdirectory(title='Select image source folder',
								initialdir=self.svSourcePath.get())
		self.svSourcePath.set(path)
		return
	def selectDesDir(self):
		path = filedialog.askdirectory(title='Select label output folder',
								initialdir=self.svDestinationPath.get())
		self.svDestinationPath.set(path)
		return
	def loadDir(self):
		self.parent.focus()
		self.imageDir = self.svSourcePath.get()
		if not os.path.isdir(self.imageDir):
			messagebox.showerror('Error!',
								message='The specified dir does not exist!')
			return
		extlist = [	'*.JPEG', '*.jpeg', '*JPG' ,
					'*.jpg' , '*.PNG' , '*.png',
					'*.BMP' , '*.bmp']
		for e in extlist:
			filelist = glob.glob(os.path.join(self.imageDir, e))
			self.imageList.extend(filelist)
		if len(self.imageList) == 0:
			print('No .JPEG images found in the specified dir!')
			return
		self.cur = 1
		self.total = len(self.imageList)
		self.outDir = self.svDestinationPath.get()
		if not os.path.exists(self.outDir): os.mkdir(self.outDir)
	def loadImage(self):
		imagepath = self.imageList[self.cur-1]
		self.img = Image.open(imagepath)
		size = self.img.size
		self.factor = max(size[0]/700, size[1]/700., 1.)
		self.img = self.img.resize((int(size[0]/self.factor),
								int(size[1]/self.factor)))
		self.tkimg = ImageTk.PhotoImage(self.img)
		self.mainPanel.config(width = max(self.tkimg.width(), 10),
								height=max(self.tkimg.height(), 10))
		self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
		self.progLabel.config(text="%04d/%04d" %(self.cur, self.total))
		self.clearBBox()
		fullfilename = os.path.basename(imagepath)
		self.imagename, _ = os.path.splitext(fullfilename)
		labelname = self.imagename + '.txt'
		self.labelfilename = os.path.join(self.outDir, labelname)
		bbox_cnt = 0
		if os.path.exists(self.labelfilename):
			with open(self.labelfilename) as f:
				for (i, line) in enumerate(f):
					if i == 0:
						bbox_cnt = int(line.strip())
						continue
					tmp = line.split()
					tmp[0] = int(int(tmp[0])/self.factor)
					tmp[1] = int(int(tmp[1])/self.factor)
					tmp[2] = int(int(tmp[2])/self.factor)
					tmp[3] = int(int(tmp[3])/self.factor)
					self.bboxList.append(tuple(tmp))
					color_index = (len(self.bboxList)-1) % len(COLORS)
					tmpId = self.mainPanel.create_rectangle(
								tmp[0], tmp[1],
								tmp[2], tmp[3],
								width = 2,
								outline = COLORS[color_index])
					self.bboxIdList.append(tmpId)
					self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)'
								%(tmp[4], tmp[0], tmp[1], tmp[2], tmp[3]))
					self.listbox.itemconfig(len(self.bboxIdList)-1,
								fg=COLORS[color_index])
	def saveImage(self):
		if self.labelfilename == '': return
		with open(self.labelfilename, 'w') as f:
			f.write('%d\n' %len(self.bboxList))
			for bbox in self.bboxList:
				f.write("{} {} {} {} {}\n".format(int(int(bbox[0])*self.factor),
								int(int(bbox[1])*self.factor),
								int(int(bbox[2])*self.factor),
								int(int(bbox[3])*self.factor),
								bbox[4]))
		print('Image No. %d saved' %(self.cur))
	def mouseClick(self, event):
		if self.STATE['click'] == 0:
			self.STATE['x'], self.STATE['y'] = event.x, event.y
		else:
			x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'],event.x)
			y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'],event.y)
			self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass))
			self.bboxIdList.append(self.bboxId)
			self.bboxId = None
			self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)'
								%(self.currentLabelclass, x1, y1, x2, y2))
			self.listbox.itemconfig(len(self.bboxIdList) - 1,
						fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
		self.STATE['click'] = 1 - self.STATE['click']
	def mouseMove(self, event):
		self.disp.config(text='x: %d, y: %d' %(event.x, event.y))
		if self.tkimg:
			if self.hl:
				self.mainPanel.delete(self.hl)
			self.hl = self.mainPanel.create_line(0, event.y,
								self.tkimg.width(), event.y, width=2)
			if self.vl:
				self.mainPanel.delete(self.vl)
			self.vl = self.mainPanel.create_line(event.x, 0,
								event.x, self.tkimg.height(), width = 2)
		if 1 == self.STATE['click']:
			if self.bboxId: self.mainPanel.delete(self.bboxId)
			COLOR_INDEX = len(self.bboxIdList) % len(COLORS)
			self.bboxId = self.mainPanel.create_rectangle(
							self.STATE['x'],
							self.STATE['y'],
							event.x, event.y,
							width=2,
							outline=COLORS[len(self.bboxList) % len(COLORS)])
	def cancelBBox(self, event):
		if 1 == self.STATE['click']:
			if self.bboxId:
				self.mainPanel.delete(self.bboxId)
				self.bboxId = None
				self.STATE['click'] = 0
	def delBBox(self):
		sel = self.listbox.curselection()
		if len(sel) != 1: return
		idx = int(sel[0])
		self.mainPanel.delete(self.bboxIdList[idx])
		self.bboxIdList.pop(idx)
		self.bboxList.pop(idx)
		self.listbox.delete(idx)
	def clearBBox(self):
		for idx in range(len(self.bboxIdList)):
			self.mainPanel.delete(self.bboxIdList[idx])
		self.listbox.delete(0, len(self.bboxList))
		self.bboxIdList = []
		self.bboxList = []
	def prevImage(self, event = None):
		self.saveImage()
		if self.cur > 1:
			self.cur -= 1
			self.loadImage()
	def nextImage(self, event = None):
		self.saveImage()
		if self.cur < self.total:
			self.cur += 1
			self.loadImage()
	def gotoImage(self):
		idx = int(self.idxEntry.get())
		if 1 <= idx and idx <= self.total:
			self.saveImage()
			self.cur = idx
			self.loadImage()
	def setClass(self):
		self.currentLabelclass = self.classcandidate.get()
		print('set label class to : {}'.format(self.currentLabelclass))

'''
MIT License

Copyright (c) 2018 Sari Sabban

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
'''

def translate(texts, images):
	''' Translates .txt annotations to .xml annotations '''
	source = 'https://github.com/sarisabban/ProtiClass'
	for thefile in os.listdir(texts):
		filename = thefile.split('.')[0]
		with open('{}/{}.xml'.format(texts, filename), 'w') as f:
			data = open('{}/{}'.format(texts, thefile), 'r')
			img = Image.open('{}/{}.jpg'.format(images, filename))
			W, H = img.size
			f.write('<annotation>\n')
			f.write('\t<filename>{}.jpg</filename>\n'.format(filename))
			f.write('\t<source>{}</source>\n'.format(source))
			f.write('\t<path>../dataset/Train/{}.jpg</path>'.format(filename))
			f.write('\t<size>\n')
			f.write('\t\t<width>{}</width>\n'.format(W))
			f.write('\t\t<height>{}</height>\n'.format(H))
			f.write('\t\t<depth>3</depth>\n')
			f.write('\t</size>\n')
			f.write('\t<segments>{}</segments>\n'.format(next(data).strip()))
			for line in data:
				line = line.split()
				xmin = line[0]
				ymin = line[1]
				xmax = line[2]
				ymax = line[3]
				label = line[4]
				f.write('\t<object>\n')
				f.write('\t\t<name>{}</name>\n'.format(label))
				f.write('\t\t<bndbox>\n')
				f.write('\t\t\t<xmin>{}</xmin>\n'.format(xmin))
				f.write('\t\t\t<ymin>{}</ymin>\n'.format(ymin))
				f.write('\t\t\t<xmax>{}</xmax>\n'.format(xmax))
				f.write('\t\t\t<ymax>{}</ymax>\n'.format(ymax))
				f.write('\t\t</bndbox>\n')
				f.write('\t</object>\n')
			f.write('</annotation>')
		print('[+] Generated file: {}.xml'.format(filename))

def txt_xml(txt_dir, img_dir):
	''' Translate and move .xml files to relevent directory '''
	translate(txt_dir, img_dir)
	os.makedirs('./dataset/Annotations', exist_ok=True)
	print('\n[+] Generated Annotations directory')
	os.system('mv ./dataset/BBox_Annotations/*.xml ./dataset/Annotations')
	print('\n[+] Moved files')
	print('-----------------------')
	print('[+] Done')

def box(text, image):
	''' Get box value from .txt file '''
	img = Image.open(image)
	text = open(text, 'r')
	next(text)
	for line in text:
		line = line.split()
		L = int(line[0])
		U = int(line[1])
		R = int(line[2])
		D = int(line[3])
		box = [L, U, R, D]
		draw = ImageDraw.Draw(img)
		draw.rectangle(box, outline='red')
	newfilename = image.split('.')[1].split('/')[-1]
	img.save('./{}_out.jpg'.format(newfilename), 'JPEG')
	print('[+] Saved file ... {}'.format(newfilename))

def check_dir():
	'''
	Check to see if the dataset directory exist
	and move files to the relevent directory
	'''
	count = 0
	for Afile in os.listdir('./dataset/Train'):
		Afile = Afile.split('.')[0]
		file_img = './dataset/Train/{}.jpg'.format(Afile)
		file_txt = './dataset/BBox_Annotations/{}.txt'.format(Afile)
		box(file_txt, file_img)
		count += 1
	print('\n[+] Total of {} files'.format(count))
	os.makedirs('./dataset/Check', exist_ok=True)
	print('\n[+] Generated Check directory')
	os.system('mv ./*.jpg ./dataset/Check')
	print('\n[+] Moved files')
	print('-----------------------')
	print('[+] Done')

def rename(Old, New):
	''' Rename a label in the whole dataset '''
	directory = './dataset/BBox_Annotations'
	for afile in os.listdir(directory):
		data_in = open('{}/{}'.format(directory, afile), 'r')
		next(data_in)
		count = 0
		lines = []
		for line in data_in:
			count += 1
			line = line.split()
			if line[-1] == Old: line[-1] = New
			comb = ' '.join(line)+'\n'
			lines.append(comb)
		print(count)
		print(lines)
		data_out = open(afile, 'w')
		data_out.write('{}\n'.format(str(count)))
		for i in lines: data_out.write(i)
		data_out.close()

def BOX(BBOX_line1, BBOX_line2):
	''' Compair two bounding boxes '''
	line1 = BBOX_line1
	line1 = line1.strip().split()
	xmin1 =int(line1[0])
	ymin1 =int(line1[1])
	xmax1 =int(line1[2])
	ymax1 =int(line1[3])
	label1= line1[4]
	line2 = BBOX_line2
	line2 = line2.strip().split()
	xmin2 =int(line2[0])
	ymin2 =int(line2[1])
	xmax2 =int(line2[2])
	ymax2 =int(line2[3])
	label2= line2[4]
	bb1 = {'x1':xmin1, 'x2':xmax1, 'y1':ymin1, 'y2':ymax1}
	bb2 = {'x1':xmin2, 'x2':xmax2, 'y1':ymin2, 'y2':ymax2}
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])
	if x_right < x_left or y_bottom < y_top: return(False)
	int_area = (x_right - x_left) * (y_bottom - y_top)
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
	IOU = round((int_area / float(bb1_area + bb2_area - int_area)), 3)
	if IOU > 0.5 and label1 == label2:
		print(IOU, label1, label2, 'True')
		return(True)
	else: return(False)

def eval(dir_test, dir_pred):
	'''
	Evaluates the Test set annotations against the network's
	predictions
	'''
	for fT, fP in zip(os.listdir(dir_test), os.listdir(dir_pred)):
		FileT = open('{}/{}'.format(dir_test, fT), 'r')
		next(FileT)
		FileP = open('{}/{}'.format(dir_pred, fP), 'r')
		for lineT in FileT:
			FileP.seek(0)
			next(FileP)
			for lineP in FileP:
				T = lineT.strip()
				P = lineP.strip()
				BOX(lineT, lineP)

def augment(input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=10):
	''' Augments images and saves them into a new directory '''
	os.makedirs('./dataset/Augmented', exist_ok=True)
	for Image in os.listdir(input_path):
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

def CNN(choice='predict', CNN='VGG16', prediction='./image.jpg'):
	''' Train images using one of several CNNs '''
	if choice == 'train':
		Train   = './dataset/Train'
		Tests   = './dataset/Test'
		Valid   = './dataset/Valid'
		shape   = (224, 224)
		epochs  = 10
		batches = 16
		classes = []
		for c in os.listdir(Train): classes.append(c)
		IDG = keras.preprocessing.image.ImageDataGenerator(
			featurewise_center=True,
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
			#interpolation_order=1,
			dtype='float32')
		train = IDG.flow_from_directory(Train, target_size=shape,
						color_mode='rgb', classes=classes, batch_size=batches)
		tests = IDG.flow_from_directory(Tests, target_size=shape,
						color_mode='rgb', classes=classes, batch_size=batches)
		valid = IDG.flow_from_directory(Valid, target_size=shape,
						color_mode='rgb', classes=classes, batch_size=batches)
		input_shape = train.image_shape
		if CNN == 'VGG16' or 'vgg16':
			model = VGG16(weights=None, input_shape=input_shape,
						classes=len(classes))
		elif CNN == 'VGG19' or 'vgg19':
			model = VGG19(weights=None, input_shape=input_shape,
						classes=len(classes))
		elif CNN == 'ResNet50' or 'resnet50':
			model = ResNet50(weights=None, input_shape=input_shape,
						classes=len(classes))
		elif CNN == 'DenseNet201' or 'densenet201':
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
		plt.save()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.save()
		Y_pred = model.predict_generator(tests, verbose=1)
		y_pred = np.argmax(Y_pred, axis=1)
		matrix = confusion_matrix(tests.classes, y_pred)
		df_cm  = pd.DataFrame(matrix, index=classes, columns=classes)
		plt.figure(figsize=(10,7))
		sn.heatmap(df_cm, annot=True)
		print(classification_report(tests.classes,y_pred,target_names=classes))
		model.save_weights('weights.h5')
	elif choice == 'predict':
		model.load_weights('weights.h5')
		prdct = model.predict(prediction)
		print(prdct)

def main():
	if args.bbox:
		P1 = 'Enter label and press enter to enter a new label.\n'
		P2 = 'Type `end` to end label entry and continue to annotation.'
		print(P1+P2)
		print('-----')
		LABELS = []
		while True:
			L = input('Input Label>')
			if L == 'end' or L == 'End' or L == 'END': break
			else: LABELS.append(L)
		print('Labels are:', LABELS)
		root = Tk()
		tool = LabelTool(root, LABELS)
		root.resizable(width=True, height=True)
		root.mainloop()
	elif args.translate:txt_xml('./dataset/BBox_Annotations','./dataset/Train')
	elif args.check: check_dir()
	elif args.rename: rename(sys.argv[2], sys.argv[3])
	elif args.eval: eval('BBox_Test', 'BBox_Test_Predictions')
	elif args.augment: augment()
	elif args.cnn_train: CNN(choice='predict', CNN=sys.argv[2])
	elif args.cnn_predict: CNN(CNN=sys.argv[2], prediction=sys.argv[3])

if __name__ == '__main__': main()
