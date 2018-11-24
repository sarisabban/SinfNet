#!/usr/bin/env python3

import os
import sys
from PIL import Image
from PIL import ImageDraw

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

def main():
	if sys.argv[1] == '-t':
		txt_xml('./dataset/BBox_Annotations', './dataset/Images')
	elif sys.argv[1] == '-cd':
		check_dir()

if __name__ == '__main__': main()
