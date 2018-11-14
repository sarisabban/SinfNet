#!/usr/bin/env python3

import os
import sys
from PIL import Image
from PIL import ImageDraw

def translate(texts, images):
	source = 'https://github.com/sarisabban/plankclass'
	for thefile in os.listdir(texts):
		filename = thefile.split('.')[0]
		with open('{}{}.xml'.format(texts, filename), 'w') as f:
			data = open('{}{}'.format(texts, thefile), 'r')
			img = Image.open('{}{}.jpg'.format(images, filename))
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
	print('[+] Done')

def txt_xml(txt_dir, img_dir):
	translate(txt_dir, img_dir)
	user = input('Delete all .txt files? (yes/no)\n>')
	if user == 'y' or user == 'yes' or user == 'Y' or user == 'YES':
		os.system('rm {}/*.txt'.format(txt_dir))
	elif user == 'n' or user == 'no' or user == 'N' or user == 'NO':
		exit()
	else:
		print('[-] Error: wrong choice, type y/n or yes/no')

def box(image, text, out):
	img = Image.open(image)
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
	newfilename = image.split('.')[0].split('/')[-1]
	img.save('{}/{}_out.jpg'.format(out, newfilename), 'JPEG')
	print('[+] Saved file ... {}'.format(newfilename))

def check_dir(directory, output):
	location = '{}/check'.format(output)
	os.mkdir(location)
	count = 0
	for Afile in os.listdir('{}/images'.format(directory)):
		Afile = Afile.split('.')[0]
		file_img = '{}/images/{}.jpg'.format(directory, Afile)
		file_txt = open('{}/annotations/{}.txt'.format(directory, Afile), 'r')
		box(file_img, file_txt, location)
		count += 1
	print('[+] Total of {} files'.format(count))

def check_file(image, text):
	filetext = open(text, 'r')
	box(image, filetext, './')

def main():
	if sys.argv[1] == '-t':
				# Text Dir	# Image Dir
		txt_xml(sys.argv[2], sys.argv[3])
	elif sys.argv[1] == '-cf':
					#Image File	#Text File
		check_file(sys.argv[2], sys.argv[3])
	elif sys.argv[1] == '-cd':
					#Dataset Dir
		check_dir(sys.argv[2], sys.argv[2])

if __name__ == '__main__': main()
