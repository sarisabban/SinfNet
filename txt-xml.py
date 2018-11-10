#!/usr/bin/env python3

import os
import sys
from PIL import Image

def translate(texts, images):
	source = 'https://github.com/sarisabban/plankclass'
	for thefile in os.listdir(texts):
		filename = thefile.split('.')[0]
		with open('{}/{}.xml'.format(texts, filename), 'w') as f:
			data = open('{}/{}'.format(texts, thefile), 'r')
			img = Image.open('{}/{}.jpg'.format(images, filename))
			W, H = img.size
			f.write('<annotation>\n')
			f.write('\t<filename>{}.jpg</filename>\n'.format(filename))
			f.write('\t<source>{}</source>\n'.format(source))
			f.write('\t<size>\n')
			f.write('\t\t<width>{}</width>\n'.format(W))
			f.write('\t\t<height>{}</width>\n'.format(H))
			f.write('\t\t<depth>3</depth>\n')
			f.write('\t</size>\n')
			f.write('\t<segments>{}</segments>\n'.format(next(data).strip()))
			for line in data:
				line	= line.split()
				xmin	= line[0]
				ymin	= line[1]
				xmax	= line[2]
				ymax	= line[3]
				label	= line[4]
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
	print('-----\nDone')

def main():
	txt_dir = sys.argv[1]
	img_dir = sys.argv[2]
	translate(txt_dir, img_dir)
	user = input('Delete all .txt files? (yes/no)\n>')
	if user == 'y' or user == 'yes' or user == 'Y' or user == 'YES':
		os.system('rm {}/*.txt'.format(txt_dir))
	elif user == 'n' or user == 'no' or user == 'N' or user == 'NO':
		exit()
	else:
		print('[-] Error: wrong choice, type y/n or yes/no')

if __name__ == '__main__': main()
