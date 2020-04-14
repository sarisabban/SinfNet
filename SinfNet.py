#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser(description='Collection of datasets and networks for organism classification')
parser.add_argument('-ct', '--cnn_train',    nargs='+', help='Train on CNN')
parser.add_argument('-cp', '--cnn_predict',  nargs='+', help='Predict from CNN')
parser.add_argument('-yt', '--yolo_train',   nargs='+', help='Train on YOLOv3')
parser.add_argument('-yp', '--yolo_predict', nargs='+', help='Predict from YOLOv3')
parser.add_argument('-c' , '--convert',      nargs='+', help='Convert Bash terminal output to .txt files')
parser.add_argument('-tx', '--translate_xml',action='store_true', help='Translate .txt file to .xml file')
parser.add_argument('-tc', '--translate_csv',action='store_true', help='Translate .csv file to .xml file')
parser.add_argument('-a' , '--augment',      action='store_true', help='Augments images')
parser.add_argument('-v' , '--via',          action='store_true', help='Open the VIA image labeling tool')
parser.add_argument('-b' , '--bbox',         action='store_true', help='Open the BBox image labeling tool')
args = parser.parse_args()

def translate_txt_xml(texts, location):
	''' Translates .txt annotations to .xml annotations '''
	source = 'https://github.com/sarisabban/SinfNet'
	for thefile in os.listdir(texts):
		filename = thefile.split('.')[0]
		with open('{}/{}.xml'.format(texts, filename), 'w') as f:
			data = open('{}/{}'.format(texts, thefile), 'r')
			img = Image.open('{}/{}.jpg'.format(location, filename))
			W, H = img.size
			f.write('<annotation>\n')
			f.write('\t<filename>{}.jpg</filename>\n'.format(filename))
			f.write('\t<source>{}</source>\n'.format(source))
			f.write('\t<path>../dataset/Train/{}.jpg</path>\n'.format(filename))
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
	os.system('mv ./dataset/BBox_Annotations/*.xml ./dataset/Annotations')
	print('\n[+] Moved files')
	print('-----------------------')
	print('[+] Done')

def translate_csv_xml(CSV, location):
	''' Translates .csv annotations to .xml annotations '''
	source = 'https://github.com/sarisabban/SinfNet'
	with open(CSV, 'r') as f:
		next(f)
		for line in f:
			line = line.strip().split(',')
			with open('{}.xml'.format(line[0].split('.')[0]), 'a') as F:
				total = line[3]
				x = line[6].split(':')[-1]
				y = line[7].split(':')[-1]
				w = line[8].split(':')[-1]
				h = line[9].split(':')[-1].split('}')[0]
				label = line[10].split(':')[-1].split('"')[2]
				l = '{} {} {} {} {}\n'.format(x, y, w, h, label)
				with open('{}.xml'.format(line[0].split('.')[0]), 'r') as t:
					img = Image.open('{}/{}'.format(location, line[0]))
					W, H = img.size
					first_line = t.readline()
					if first_line != '<annotation>\n':
						F.write('<annotation>\n')
						F.write('\t<filename>{}</filename>\n'.format(line[0]))
						F.write('\t<source>{}</source>\n'.format(source))
						F.write('\t<path>../dataset/Train/{}</path>\n'\
						.format(line[0]))
						F.write('\t<size>\n')
						F.write('\t\t<width>{}</width>\n'.format(W))
						F.write('\t\t<height>{}</height>\n'.format(H))
						F.write('\t\t<depth>3</depth>\n')
						F.write('\t</size>\n')
						F.write('\t<segments>{}</segments>\n'.format(total))
						print('[+] Generated file: {}.xml'\
						.format(line[0].split('.')[0]))
					else: pass
				F.write('\t<object>\n')
				F.write('\t\t<name>{}</name>\n'.format(label))
				F.write('\t\t<bndbox>\n')
				F.write('\t\t\t<xmin>{}</xmin>\n'.format(x))
				F.write('\t\t\t<ymin>{}</ymin>\n'.format(y))
				F.write('\t\t\t<xmax>{}</xmax>\n'.format(w))
				F.write('\t\t\t<ymax>{}</ymax>\n'.format(h))
				F.write('\t\t</bndbox>\n')
				F.write('\t</object>\n')
	os.system('mv ./*.xml ./dataset/Annotations')
	print('-----------------------')
	print('[+] Moved files')
	for item in os.listdir('./dataset/Annotations'):
		with open('./dataset/Annotations/{}'.format(item), 'a') as c:
			c.write('</annotation>\n')
	print('[+] XML endings added')
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

def main():
	if args.cnn_train:
		from sources import CNN
		CNN.CNN(sys.argv[2], 'train', '', '')
	elif args.cnn_predict:
		from sources import CNN
		CNN.CNN(sys.argv[2], 'predict', sys.argv[3] ,sys.argv[4])
	elif args.via:
		os.system('firefox ./sources/VIA.html')
	elif args.yolo_train:
		from sources import YOLOv3
		YOLOv3.main_train()
	elif args.yolo_predict:
		from sources import YOLOv3
		YOLOv3.main_predict(sys.argv[2], sys.argv[3], './')
	elif args.translate_xml:
		translate_txt_xml('./dataset/BBox_Annotations', './dataset/Train')
	elif args.translate_csv:
		translate_csv_xml('via_export_csv.csv', './dataset/Train')
	elif args.convert:
		convert(sys.argv[2])
	elif args.augment:
		from sources import Augment
		Augment.augment(
			input_path='./dataset/Train',
			output_path='./dataset/Augmented',
			count=sys.argv[2])
	elif args.bbox:
		from sources import BBox
		BBox.main()

if __name__ == '__main__': main()
