#!/usr/bin/env python3

import os
import glob
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

COLORS = ['red', 'blue','pink', 'cyan', 'green', 'black'] # colors for the bboxes
SIZE = 256, 256 # image sizes for the examples

class LabelTool():
	'''
	GUI bounding box annotation tool

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

def main():
	os.makedirs('./dataset/BBox_Annotations', exist_ok=True)
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

if __name__ == '__main__': main()
