import os
import sys
import numpy
#import keras
from PIL import Image			#sudo pacman -S python-pillow

def segment(filename, name):
	os.mkdir(name)
	img = Image.open(filename)
	pix = img.getdata()
	# Image Dimentions
	x = 145
	y = 145
	# Kernel (30x30)px every 5px [MNIST 28x28]
	step = 10
	L = 0
	U = 0
	R = 30
	D = 30
	count = 1
	for total in range(3):
		for kernel in range(10):
			box = (L, U, R, D)
			region = img.crop(box)
			region.save('{}/o{}.jpg'.format(name, count), 'JPEG')
			count += 1
			R += step
			L += step
		L = 0
		U += step
		R = 30
		D += step

def sort():
	pass

def augment(directory):
	for filename in os.listdir(directory):
		# Original
		img = Image.open('{}/{}'.format(directory, filename))
		# Rotate original
		img.rotate(90).save('{}/O90{}'.format(directory, filename))
		img.rotate(180).save('{}/O180{}'.format(directory, filename))
		img.rotate(270).save('{}/O270{}'.format(directory, filename))
		# Flip
		imgFLIP = img.transpose(Image.FLIP_TOP_BOTTOM)
		imgFLIP.save('{}/F{}'.format(directory, filename))
		# Rotate flipped
		imgFLIP.rotate(90).save('{}/F90{}'.format(directory, filename))
		imgFLIP.rotate(180).save('{}/F180{}'.format(directory, filename))
		imgFLIP.rotate(270).save('{}/F270{}'.format(directory, filename))
	# Rename images
	count = 1
	for image in os.listdir(directory):
		os.rename('{}/{}'.format(directory, image), '{}/{}.jpg'.format(directory, str(count)))
		count += 1


def dataset():
	pass

def CNN():
	pass









def main():
	Microscope_Image = 'earth.jpg'
	species_Name = 'earth'
	segment(Microscope_Image, species_Name)
	augment(species_Name)



if __name__ == '__main__':
	main()
