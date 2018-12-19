#!/usr/bin/env python3

#-----------------------------------------------------------------------
'''
MIT License

Copyright (c) 2017 Shi Qiu

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

This script is modified from https://github.com/xiaqunfeng/BBox-Label-Tool
which is in turn adopted from https://github.com/puzzledqs/BBox-Label-Tool

BBox-Label-Tool
===============
A simple tool for labeling object bounding boxes in images, implemented
with Python Tkinter.

Usage
-----
0.	Add all the classes that you will be labeling in the list in line 61
1.	Run the following command from the terminal: python3 Label.py -BBox
2.	Click "Image Input Folder" on the top left to choose the directory that
	contains the images (./dataset/Images)
3.	Click "Label Output Folder" on the top left to choose the directory that
	will save the lables (./dataset/Annotations)
4.	Click "Load Dir" on the top right to load your choices (nothing will happen)
5.	Click "Next >>" on the bottom to load the second image
5.	Click "Previous >>" to go back to the first image and start annotating
6.	Label images with boxes
7.	Click "Next >>" to save labels and move on to the next image
8.	Support multiple image formats: .JPEG .jpeg .JPG .jpg .PNG .png .BMP .bmp
'''

import os
import sys
import glob
import random
import argparse
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw

# class labels
LABELS = ['Inactive', 'Active']

# colors for the bboxes
COLORS = ['red', 'blue','pink', 'cyan', 'green', 'black']

# image sizes for the examples
SIZE = 256, 256

parser = argparse.ArgumentParser(description='Object Detection Neural Network Dataset Labeling')
parser.add_argument('-b', '--bbox', action='store_true', help='Open the BBox image labeling tool')
parser.add_argument('-t', '--translate', action='store_true', help='Translate .txt file to .xml file')
parser.add_argument('-c', '--check', action='store_true', help='Check the images for correct annotation')
parser.add_argument('-r', '--rename', nargs='+', help='Rename a label')
args = parser.parse_args()

class LabelTool():
	def __init__(self, master):
		# set up the main frame
		self.parent = master
		self.parent.title("BBox Label Tool")
		self.frame = Frame(self.parent)
		self.frame.pack(fill=BOTH, expand=1)
		self.parent.resizable(width = FALSE, height = FALSE)
		# initialize global state
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
		self.cla_can_temp = LABELS#[]
		#self.classcandidate_filename = 'class.txt'
		# initialize mouse state
		self.STATE = {}
		self.STATE['click'] = 0
		self.STATE['x'], self.STATE['y'] = 0, 0
		# reference to bbox
		self.bboxIdList = []
		self.bboxId = None
		self.bboxList = []
		self.hl = None
		self.vl = None
		# ----------------- GUI stuff ---------------------
		# dir entry & load
		# input image dir button
		self.srcDirBtn = Button(self.frame, text="Image input folder", \
													command=self.selectSrcDir)
		self.srcDirBtn.grid(row=0, column=0)
		# input image dir entry
		self.svSourcePath = StringVar()
		self.entrySrc = Entry(self.frame, textvariable=self.svSourcePath)
		self.entrySrc.grid(row=0, column=1, sticky=W+E)
		self.svSourcePath.set(os.path.join(os.getcwd(),"dataset/Images"))
		# load button
		self.ldBtn = Button(self.frame, text="Load Dir", command=self.loadDir)
		self.ldBtn.grid(row=0, column=2, rowspan=2, \
						columnspan=2, padx=2, pady=2, \
						ipadx=5, ipady=5)
		# label file save dir button
		self.desDirBtn = Button(self.frame, text="Label output folder", \
													command=self.selectDesDir)
		self.desDirBtn.grid(row=1, column=0)
		# label file save dir entry
		self.svDestinationPath = StringVar()
		self.entryDes = Entry(self.frame, textvariable=self.svDestinationPath)
		self.entryDes.grid(row=1, column=1, sticky=W+E)
		self.svDestinationPath.set(os.path.join(os.getcwd(),
									"dataset/BBox_Annotations"))
		# main panel for labeling
		self.mainPanel = Canvas(self.frame, cursor='tcross')
		self.mainPanel.bind("<Button-1>", self.mouseClick)
		self.mainPanel.bind("<Motion>", self.mouseMove)
		self.parent.bind("<Escape>", self.cancelBBox)# press <Espace> to cancel
		self.parent.bind("s", self.cancelBBox)
		self.parent.bind("p", self.prevImage) # press 'p' to go backforward
		self.parent.bind("n", self.nextImage) # press 'n' to go forward
		self.mainPanel.grid(row = 2, column = 1, rowspan = 4, sticky = W+N)
		# choose class
		self.classname = StringVar()
		self.classcandidate = ttk.Combobox(self.frame, state='readonly', \
													textvariable=self.classname)
		self.classcandidate.grid(row=2, column=2)
		# Commented out so we don't require a separate class.txt file for labels
		#if os.path.exists(self.classcandidate_filename):
			#with open(self.classcandidate_filename) as cf:
				#for line in cf.readlines():
					#self.cla_can_temp.append(line.strip('\n'))
		self.classcandidate['values'] = self.cla_can_temp
		self.classcandidate.current(0)
		self.currentLabelclass = self.classcandidate.get()
		self.btnclass = Button(self.frame, text='Confirm Class', \
														command=self.setClass)
		self.btnclass.grid(row=2, column=3, sticky=W+E)
		# showing bbox info & delete bbox
		self.lb1 = Label(self.frame, text = 'Bounding boxes:')
		self.lb1.grid(row = 3, column = 2,  sticky = W+N)
		self.listbox = Listbox(self.frame, width = 22, height = 12)
		self.listbox.grid(row = 4, column = 2, sticky = N+S)
		self.btnDel = Button(self.frame, text = 'Delete', command=self.delBBox)
		self.btnDel.grid(row = 4, column = 3, sticky = W+E+N)
		self.btnClear = Button(self.frame, text = 'Clear All', \
														command=self.clearBBox)
		self.btnClear.grid(row = 4, column = 3, sticky = W+E+S)
		# control panel for image navigation
		self.ctrPanel = Frame(self.frame)
		self.ctrPanel.grid(row = 6, column = 1, columnspan = 2, sticky = W+E)
		self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, \
														command=self.prevImage)
		self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
		self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, \
														command=self.nextImage)
		self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
		self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
		self.progLabel.pack(side = LEFT, padx = 5)
		self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
		self.tmpLabel.pack(side = LEFT, padx = 5)
		self.idxEntry = Entry(self.ctrPanel, width = 5)
		self.idxEntry.pack(side = LEFT)
		self.goBtn = Button(self.ctrPanel, text = 'Go', command=self.gotoImage)
		self.goBtn.pack(side = LEFT)
		# example pannel for illustration
		#self.egPanel = Frame(self.frame, border = 10)
		#self.egPanel.grid(row = 3, column = 0, rowspan = 5, sticky = N)
		#self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
		#self.tmpLabel2.pack(side = TOP, pady = 5)
		#self.egLabels = []
		#for i in range(3):
			#self.egLabels.append(Label(self.egPanel))
			#self.egLabels[-1].pack(side = TOP)
		# display mouse position
		self.disp = Label(self.ctrPanel, text='')
		self.disp.pack(side = RIGHT)
		self.frame.columnconfigure(1, weight = 1)
		self.frame.rowconfigure(4, weight = 1)

	def selectSrcDir(self):
		path = filedialog.askdirectory(title="Select image source folder", \
											initialdir=self.svSourcePath.get())
		self.svSourcePath.set(path)
		return

	def selectDesDir(self):
		path = filedialog.askdirectory(title="Select label output folder", \
										initialdir=self.svDestinationPath.get())
		self.svDestinationPath.set(path)
		return

	def loadDir(self):
		self.parent.focus()
		# get image list
		#self.imageDir = os.path.join(r'./Images', '%03d' %(self.category))
		self.imageDir = self.svSourcePath.get()
		if not os.path.isdir(self.imageDir):
			messagebox.showerror("Error!", \
									message="The specified dir doesn't exist!")
			return

		extlist = [	"*.JPEG", "*.jpeg", "*JPG", 
					"*.jpg", "*.PNG", "*.png", 
					"*.BMP", "*.bmp"]
		for e in extlist:
			filelist = glob.glob(os.path.join(self.imageDir, e))
			self.imageList.extend(filelist)
		#self.imageList = glob.glob(os.path.join(self.imageDir, '*.JPEG'))
		if len(self.imageList) == 0:
			print('No .JPEG images found in the specified dir!')
			return

		# default to the 1st image in the collection
		self.cur = 1
		self.total = len(self.imageList)
		# set up output dir
		#self.outDir = os.path.join(r'./Labels', '%03d' %(self.category))
		self.outDir = self.svDestinationPath.get()
		if not os.path.exists(self.outDir):
			os.mkdir(self.outDir)
		# load example bboxes
		#self.egDir = os.path.join(r'./Examples', '%03d' %(self.category))
		#self.egDir = os.path.join(os.getcwd(), "Examples/001")
		#if not os.path.exists(self.egDir):
			#return
		#filelist = glob.glob(os.path.join(self.egDir, '*.JPEG'))
		#self.tmp = []
		#self.egList = []
		#random.shuffle(filelist)
		#for (i, f) in enumerate(filelist):
			#if i == 1:
				#break
			#im = Image.open(f)
			#r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
			#new_size = int(r * im.size[0]), int(r * im.size[1])
			#self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
			#self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
			#self.egLabels[i].config( \
					#image=self.egList[-1], width = SIZE[0], height = SIZE[1])
		#self.loadImage()
		#print('%d images loaded from %s' %(self.total, self.imageDir))

	def loadImage(self):
		# load image
		imagepath = self.imageList[self.cur - 1]
		self.img = Image.open(imagepath)
		size = self.img.size
		self.factor = max(size[0]/700, size[1]/700., 1.)
		self.img = self.img.resize((int(size[0]/self.factor), \
													int(size[1]/self.factor)))
		self.tkimg = ImageTk.PhotoImage(self.img)
		self.mainPanel.config(width = max(self.tkimg.width(), 10), \
										height=max(self.tkimg.height(), 10))
		self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
		self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
		# load labels
		self.clearBBox()
		#self.imagename = os.path.split(imagepath)[-1].split('.')[0]
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
					#tmp = [int(t.strip()) for t in line.split()]
					tmp = line.split()
					tmp[0] = int(int(tmp[0])/self.factor)
					tmp[1] = int(int(tmp[1])/self.factor)
					tmp[2] = int(int(tmp[2])/self.factor)
					tmp[3] = int(int(tmp[3])/self.factor)
					self.bboxList.append(tuple(tmp))
					color_index = (len(self.bboxList)-1) % len(COLORS)
					tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
															tmp[2], tmp[3], \
															width = 2, \
												outline = COLORS[color_index])
					#outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
					self.bboxIdList.append(tmpId)
					self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' \
									%(tmp[4], tmp[0], tmp[1], tmp[2], tmp[3]))
					self.listbox.itemconfig(len(self.bboxIdList) - 1, \
													fg = COLORS[color_index])
					#self.listbox.itemconfig(len(self.bboxIdList) - 1, \
					#fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

	def saveImage(self):
		if self.labelfilename == '':
			return
		with open(self.labelfilename, 'w') as f:
			f.write('%d\n' %len(self.bboxList))
			for bbox in self.bboxList:
				f.write("{} {} {} {} {}\n".format(int(int(bbox[0])*self.factor),
												int(int(bbox[1])*self.factor),
												int(int(bbox[2])*self.factor),
												int(int(bbox[3])*self.factor),
												bbox[4]))
				#f.write(' '.join(map(str, bbox)) + '\n')
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
			self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' \
									%(self.currentLabelclass, x1, y1, x2, y2))
			self.listbox.itemconfig(len(self.bboxIdList) - 1, \
						fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
		self.STATE['click'] = 1 - self.STATE['click']

	def mouseMove(self, event):
		self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
		if self.tkimg:
			if self.hl:
				self.mainPanel.delete(self.hl)
			self.hl = self.mainPanel.create_line(0, event.y, \
										self.tkimg.width(), event.y, width = 2)
			if self.vl:
				self.mainPanel.delete(self.vl)
			self.vl = self.mainPanel.create_line(event.x, 0, \
										event.x, self.tkimg.height(), width = 2)
		if 1 == self.STATE['click']:
			if self.bboxId:
				self.mainPanel.delete(self.bboxId)
			COLOR_INDEX = len(self.bboxIdList) % len(COLORS)
			self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], \
															self.STATE['y'], \
															event.x, event.y, \
															width = 2, \
							outline = COLORS[len(self.bboxList) % len(COLORS)])

	def cancelBBox(self, event):
		if 1 == self.STATE['click']:
			if self.bboxId:
				self.mainPanel.delete(self.bboxId)
				self.bboxId = None
				self.STATE['click'] = 0

	def delBBox(self):
		sel = self.listbox.curselection()
		if len(sel) != 1 :
			return
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
		print('set label class to : %s' % self.currentLabelclass)
#-----------------------------------------------------------------------
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
			f.write('\t<path>../dataset/images/{}.jpg</path>'.format(filename))
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
	translate(txt_dir, img_dir)
	os.makedirs('./dataset/Annotations', exist_ok=True)
	print('\n[+] Generated Annotations directory')
	os.system('mv ./dataset/BBox_Annotations/*.xml ./dataset/Annotations')
	print('\n[+] Moved files')
	print('-----------------------')
	print('[+] Done')

def box(text, image):
	img = Image.open(image)
	text = open(text, 'r')
	next(text)
	for line in text:
		line = line.split()
		L = int(line[0])
		U = int(line[1])
		R = int(line[2])
		D = int(line[3])
		# Box location
		box = [L, U, R, D]
		draw = ImageDraw.Draw(img)
		# Add box to image
		draw.rectangle(box, outline='red')
		# Export image
	newfilename = image.split('.')[1].split('/')[-1]
	img.save('./{}_out.jpg'.format(newfilename), 'JPEG')
	print('[+] Saved file ... {}'.format(newfilename))

def check_dir():
	count = 0
	for Afile in os.listdir('./dataset/Images'):
		Afile = Afile.split('.')[0]
		file_img = './dataset/Images/{}.jpg'.format(Afile)
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
	'''Rename a label'''
	directory = './dataset/BBox_Annotations'
	for afile in os.listdir(directory):
		data_in = open('{}/{}'.format(directory, afile), 'r')
		next(data_in)
		count = 0
		lines = []
		for line in data_in:
			count += 1
			line = line.split()
			if line[-1] == Old:
				line[-1] = New
			comb = ' '.join(line)+'\n'
			lines.append(comb)
		print(count)
		print(lines)
		data_out = open(afile, 'w')
		data_out.write('{}\n'.format(str(count)))
		for i in lines:
			data_out.write(i)
		data_out.close()

def main():
	if args.bbox:
		root = Tk()
		tool = LabelTool(root)
		root.resizable(width=True, height=True)
		root.mainloop()
	elif args.translate:
		txt_xml('./dataset/BBox_Annotations', './dataset/Images')
	elif args.check:
		check_dir()
	elif args.rename:
		rename(sys.argv[2], sys.argv[3])

if __name__ == '__main__': main()
