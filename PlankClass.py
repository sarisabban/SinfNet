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
			region.save('{}/{}.jpeg'.format(name, count), 'JPEG')
			count += 1
			R += step
			L += step
		L = 0
		U += step
		R = 30
		D += step

def sort():
	pass

def augment():
	pass

def dataset():
	pass

def CNN():
	pass









def main():
	Microscope_Image = 'earth.jpg'
	species_Name = 'earth'

	segment(Microscope_Image, species_Name)

if __name__ == '__main__':
	main()
