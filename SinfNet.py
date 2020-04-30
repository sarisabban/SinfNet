#!/usr/bin/env python3

import os
import sys
import csv
import math
import json
import shutil
import datetime
import argparse
from PIL import Image
from collections import defaultdict

parser = argparse.ArgumentParser(description='Collection of datasets and networks for organism classification')
parser.add_argument('-ct', '--cnn_train',       nargs='+', help='Train on CNN')
parser.add_argument('-cp', '--cnn_predict',     nargs='+', help='Predict from CNN')
parser.add_argument('-yt', '--yolo_train',      nargs='+', help='Train on YOLOv3')
parser.add_argument('-yp', '--yolo_predict',    nargs='+', help='Predict from YOLOv3')
parser.add_argument('-st', '--semantic_train',  nargs='+', help='Train on UNet')
parser.add_argument('-sp', '--semantic_predict',nargs='+', help='Predict from UNet')
parser.add_argument('-c' , '--convert',         nargs='+', help='Convert Bash terminal output to .txt files')
parser.add_argument('-tb', '--translate_bbox',  nargs='+', help='Translate between bounding box annotation file formats')
parser.add_argument('-tp', '--translate_poly',  nargs='+', help='Translate between polygon annotation file formats')
parser.add_argument('-ap', '--augment_poly',    nargs='+', help='Augment images with bounding polygons')
parser.add_argument('-ab', '--augment_bbox',    nargs='+', help='Augment images with bounding boxes')
parser.add_argument('-C', '--crop',             nargs='+', help='Crop images to make their dimentions multiples of 32')
parser.add_argument('-a' , '--augment',         action='store_true', help='Augment images')
parser.add_argument('-v' , '--via',             action='store_true', help='Open the VIA image labeling tool')
parser.add_argument('-b' , '--bbox',            action='store_true', help='Open the BBox image labeling tool')
args = parser.parse_args()

def translate_bbox(	image_path='./dataset/Train',
					ann_input='./dataset/Annotations',
					ann_output='./dataset/Augmented_Annotations',
					input_format='txt',
					output_format='txt'):
	''' Translating between different bounding box annotation formats '''
	if input_format == 'txt':
		BBOX = defaultdict(list)
		for filename in os.listdir(ann_input):
			with open('{}/{}'.format(ann_input, filename), 'r') as f:
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
		with open(ann_input, 'r') as F:
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
		for item in os.listdir(ann_input):
			with open('{}/{}'.format(ann_input, item), 'r') as F:
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
	elif input_format == 'coco':
		print('Not Currently Implemented')
	if output_format == 'txt':
		for name in BBOX:
			filename = name.split('.')[0]
			output = '{}/{}.txt'.format(ann_output, filename)
			with open(output, 'w') as f:
				total = str(len(BBOX[name]))
				f.write(total+'\n')
				for line in BBOX[name]:
					line = ' '.join(str(v) for v in line)+'\n'
					f.write(line)
	elif output_format == 'csv':
		output = '{}/Translated.csv'.format(ann_output)
		for name in BBOX:
			with open(output, 'a+') as f:
				header = 'filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n'
				line = f.seek(0)
				if f.readline() != header:
					f.write(header)
				items = 0
				for line in BBOX[name]:
					line = [str(i) for i in line]
					x = line[0]
					y = line[1]
					w = line[2]
					h = line[3]
					label = line[4]
					total = str(len(BBOX[name]))
					size=os.stat('./{}/{}'.format(image_path, name)).st_size
					items += 1
					TheLine = '{},{},"{{}}",{},{},"{{""name"":""rect"",""x"":{},""y"":{},""width"":{},""height"":{}}}","{{""{}"":""""}}"\n'\
					.format(filename, size, total, items, x, y, w, h, label)
					f.write(TheLine)
	elif output_format == 'xml':
		for name in BBOX:
			filename = name.split('.')[0]
			output = '{}/{}.xml'.format(ann_output, filename)
			with open(output, 'w') as f:
					source = 'https://github.com/sarisabban/SinfNet'
					total = str(len(BBOX[name]))
					W, H = Image.open('{}/{}'.format(image_path, name)).size
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
					for line in BBOX[name]:
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
	elif output_format == 'coco':
		output = '{}/temp'.format(ann_output)
		with open(output, 'w') as f:
			year = datetime.datetime.now().strftime('%Y')
			time = datetime.datetime.now().strftime('%A %d %B %Y @ %H:%M')
			header = '{{"info":\n\t\t{{"year":{},\n\t\t"version":"1",\n\t\t"description":"Microorganism Annotation",\n\t\t"contributor":"",\n\t\t"url":"https://github.com/sarisabban/sinfnet",\n\t\t"date_created":"{}"}},\n\n"images":\n'.format(year, time)
			f.write(header)
			IMG_LIST = []
			img_count = 0
			for name in BBOX:
				W, H = Image.open('{}/{}'.format(image_path, name)).size
				The_img_Line = '{{"id":{},"width":{},"height":{},"file_name":"{}","license":1,"date_captured":""}}'\
				.format(img_count, W, H, name)
				IMG_LIST.append(The_img_Line)
				img_count += 1
			f.write('\t\t'+str(IMG_LIST))
			f.write(',')
			ANN_LIST=[]
			f.write('\n\t\t\n"annotations":\n')
			img_count = 0
			ann_count = 0
			labels = []
			for name in BBOX:
				for i in BBOX[name]: labels.append(i[-1])
			collect = set(labels)
			code = {}
			codeR={}
			for l, n in enumerate(collect):
				code[n] = l
				codeR[l]= n
			for name in BBOX:
				size=str(os.stat('./{}/{}'.format(image_path, name)).st_size)
				ID = name+size
				for line in BBOX[name]:
					line = [str(i) for i in line]
					x = line[0]
					y = line[1]
					w = line[2]
					h = line[3]
					area = str(int(w)*int(h))
					label = line[4]
					The_ann_Line = '{{"id":{},"image_id":"{}","segmentation":[{},{},{},{},{},{},{},{}],"area":{},"bbox":[{},{},{},{}],"iscrowd":0, "categoty_id":{}}}'\
					.format(ann_count, img_count, x, y, str(int(x)+int(w)), y, str(int(x)+int(w)), str(int(y)+int(h)), x, str(int(y)+int(h)), area, x, y, w, h, code[label])
					ANN_LIST.append(The_ann_Line)
					ann_count += 1
				img_count += 1
			f.write('\t\t'+str(ANN_LIST))
			f.write(',')
			f.write('\n\t\t\n"categories":\n')
			CAT_LIST = []
			for labs in code:
				The_cat_Line = '{{"id":{}, "name":"{}"}}'\
				.format(code[labs], codeR[code[labs]])
				CAT_LIST.append(The_cat_Line)
			f.write('\t\t'+str(CAT_LIST))
			f.write(',')
			f.write('\n\n')
			f.write('"licenses":[{"id":1,"name":"MIT","url":"https://github.com/sarisabban/SinfNet/blob/master/LICENSE"}]}')
		with open('{}/temp'.format(ann_output), 'rt') as fin:
			with open('{}/Translated.json'.format(ann_output), 'wt') as fout:
				for line in fin:
					fout.write(line.replace("'", ""))
		os.remove('{}/temp'.format(ann_output))
	print('[+] Done')

def translate_poly(	image_path='./dataset/Train',
					ann_input='./dataset/Annotations',
					ann_output='./dataset/Augmented_Annotations',
					input_format='csv',
					output_format='json'):
	''' Translating between different polygon annotation formats '''
	if input_format == 'csv':
		TheLines = []
		POLY = defaultdict(list)
		with open(ann_input, 'r') as F:
			next(F)
			for line in F:
				line = line.strip().split(':')
				filename = line[0].split(',')[0]
				path = '{}/{}'.format(image_path, filename)
				W, H = Image.open(path).size
				lineColor = [0, 255, 0, 128]
				fillColor = [255, 255, 0, 128]
				shape_type = line[1].split('"')[2]
				line_color = 'null'
				fill_color = 'null'
				label = line[3].split('"')[-3]
				x_points = line[2].split('"')[0][:-2][1:].split(',')
				y_points = line[3].split('"')[0][:-2][1:].split(',')
				points = []
				for x, y in zip(x_points, y_points):
					try:
						x = int(x)
						y = int(y)
						point = [x, y]
						points.append(point)
					except:
						x = float(x)
						y = float(y)
						point = [x, y]
						points.append(point)

				POLY[filename,
					str(lineColor),
					str(fillColor),
					path, W, H].append([label,
										line_color,
										fill_color,
										points,
										shape_type])
	elif input_format == 'json':
		POLY = defaultdict(list)
		for TheFile in os.listdir(ann_input):
			with open('{}/{}'.format(ann_input, TheFile), 'r') as f:
				d = json.load(f)
				filename = TheFile.split('.')[0]+'.jpg'
				W, H = d['imageWidth'], d['imageHeight']
				lineColor = d['lineColor']
				fillColor = d['fillColor']
				path = d['imagePath']
				size = os.stat('{}/{}'.format(image_path, filename)).st_size
				for tot, x in enumerate(d['shapes']): total = tot+1
				num = 0
				for item in d['shapes']:
					label = item['label']
					points = item['points']
					x = []
					y = []
					for i in points:
						x.append(i[0])
						y.append(i[1])
					shape_type = item['shape_type']
					line_color = item['line_color']
					fill_color = item['fill_color']
					POLY[filename,
						str(lineColor),
						str(fillColor),
						path, W, H].append([label,
											line_color,
											fill_color,
											points,
											shape_type])
					num += 1
	if output_format == 'csv':
		with open('{}/Translated.csv'.format(ann_output), 'w+') as f:
			header = 'filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n'
			line = f.seek(0)
			if f.readline() != header:
				f.write(header)
			for name in POLY:
				filename = name[0]
				size = os.stat('{}/{}'.format(image_path, filename)).st_size
				for tot, x in enumerate(POLY[name]): total = tot+1
				num = 0
				for item in POLY[name]:
					shapetype = item[4]
					label = item[0]
					x = []
					y = []
					for points in item[3]:
						x.append(points[0])
						y.append(points[1])
					TheLine = '{},{},"{{}}",{},{},"{{""name"":""{}"",""all_points_x"":{},""all_points_y"":{}}}","{{""{}"":""""}}"\n'\
					.format(filename, size, total, num, shape_type, x, y, label)
					f.write(TheLine)
					num += 1
	elif output_format == 'json':
		for name in POLY:
			filename = name[0].split('.')[0]
			with open('{}/{}.json'.format(ann_output, filename), 'w+') as f:
				version = '3.11.2'
				flags = ''
				lineColor = name[1]
				fillColor = name[2]
				path = name[3]
				imageData = ''
				W, H = str(name[4]), str(name[5])
				header = '{{"version": "{}",\n"flags": {{{}}},\n"lineColor": {},\n"fillColor": {},\n"imagePath": "{}",\n"imageData": "{}",\n"imageHeight": {},\n"imageWidth": {},\n"shapes": ['\
				.format(version, flags, lineColor, fillColor, path, imageData, W, H)
				f.write(header)
				for info in POLY[name]:
					shape_type = info[4]
					line_color = info[1]
					fill_color = info[2]
					label = info[0]
					points = info[3]
					body = '\n\t{{"label": "{}",\n\t\t"line_color": {},\n\t\t"fill_color": {},\n\t\t"points": {},\n\t\t"shape_type": "{}"}},'\
					.format(label, line_color, fill_color, points, shape_type)
					f.write(body)
				loc = f.seek(0, os.SEEK_END)
				f.seek(loc-1)
				f.write(']}')
	print('[+] Done')

def convert(directory):
	''' Converts Bash terminal output to .txt file for Cell auto detection '''
	Items = []
	temp = None
	with open(directory, 'r') as f:
		count = 0
		for line in f:
			line = line.strip().split()
			if line == []: pass
			elif directory in line[0].split('/'):
				Items.append(temp)
				temp = []
				name = line[0].split('/')[-1]
				count = 0
				temp.append(name.split('.')[0])
			else:
				coord = line[:4]
				coord.append(directory)
				coord = ' '.join(coord)
				count += 1
				temp.append('{}\n'.format(coord))
		Items.append(temp)
	Items = Items[1:]
	for item in Items:
		name = '{}.txt'.format(item[0])
		coords = item[1:]
		coords = ''.join(coords)
		count = '{}\n'.format(len(item)-1)
		with open(name, 'w') as F:
			F.write(count)
			F.write(coords)
		print('[+] Completed {}'.format(name))
	os.makedirs('./dataset/BBox_Annotations', exist_ok=True)
	os.system('mv *.txt ./dataset/BBox_Annotations')
	translate_txt_xml('./dataset/BBox_Annotations', './dataset/Train')
	os.system('rm -r ./dataset/BBox_Annotations')

def crop(directory):
	''' Crops images to make their dimetions multiples of 32'''
	os.makedirs('./dataset/Cropped', exist_ok=True)
	for i in os.listdir(directory):
		filename = '{}/{}'.format(directory, i)
		img = Image.open(filename)
		W, H = img.size
		w = 32* math.floor(W/32)
		h = 32* math.floor(H/32)
		area = (0, 0, w, h)
		c_img = img.crop(area)
		c_img.save('./dataset/Cropped/{}'.format(i))

def main():
	if args.cnn_train:
		from sources import CNN
		CNN.CNN(sys.argv[2], 'train', '', '')
	elif args.cnn_predict:
		from sources import CNN
		CNN.CNN(sys.argv[2], 'predict', sys.argv[3], sys.argv[4])
	elif args.via:
		os.system('firefox ./sources/VIA.html')
	elif args.yolo_train:
		from sources import YOLOv3
		YOLOv3.main_train()
	elif args.yolo_predict:
		from sources import YOLOv3
		YOLOv3.main_predict(sys.argv[2], sys.argv[3], './')
	elif args.translate_bbox:
		translate_bbox(	image_path=sys.argv[2],
						ann_input=sys.argv[3],
						ann_output=sys.argv[4],
						input_format=sys.argv[5],
						output_format=sys.argv[6])
	elif args.translate_poly:
		translate_poly(	image_path=sys.argv[2],
						ann_input=sys.argv[3],
						ann_output=sys.argv[4],
						input_format=sys.argv[5],
						output_format=sys.argv[6])
	elif args.convert:
		convert(sys.argv[2])
	elif args.augment:
		from sources import Augment
		Augment.augment(
			input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=sys.argv[2])
	elif args.augment_poly:
		from sources import Augment
		Ann_in = sys.argv[2]
		iters = sys.argv[3]
		Fname = Ann_in.split('/')[-1].split('.')[0]
		os.makedirs('./dataset/Augmented', exist_ok=True)
		os.makedirs('./dataset/Augmented_Annotations', exist_ok=True)
		os.makedirs('./dataset/Translated_Annotations', exist_ok=True)
		translate_poly(	image_path='./dataset/Train',
						ann_input=Ann_in,
						ann_output='./dataset/Translated_Annotations',
						input_format='csv',
						output_format='json')
		for Images in os.listdir('./dataset/Train'):
			Augment.augment_poly('./dataset/Train/{}'.format(Images),
								'./dataset/Augmented',
								'./dataset/Translated_Annotations/{}.json'.format(Images.split('.')[0]),
								'./dataset/Augmented_Annotations', 
								iters)
		translate_poly(	image_path='./dataset/Augmented',
						ann_input='./dataset/Augmented_Annotations',
						ann_output='./dataset',
						input_format='json',
						output_format='csv')
		os.rename(	'./dataset/Translated.csv',
					'./dataset/Aug_{}.csv'.format(Fname))
		shutil.rmtree('./dataset/Translated_Annotations')
		shutil.rmtree('./dataset/Augmented_Annotations')
	elif args.augment_bbox:
		from sources import Augment
		augment_bbox(image_input='./dataset/Train',
			image_output='./dataset/Augmented',
			bbox_input='./dataset/Annotations',
			bbox_output='./dataset/Augmented_Annotations',
			count=sys.argv[2],
			input_format=sys.argv[3],
			output_format=sys.argv[4])
	elif args.bbox:
		from sources import BBox
		BBox.main()
	elif args.semantic_train:
		from sources import semantic
		semantic.semantic_train()
	elif args.semantic_predict:
		from sources import semantic
		semantic.semantic_predict(sys.argv[2])
	elif args.crop:
		crop(sys.argv[2])

if __name__ == '__main__': main()
