import os
import sys
from PIL import Image
from PIL import ImageDraw

def box(filename):
	img = Image.open(filename)
	pix = img.getdata()
	# Box location
	L = 78
	U = 336
	R = 184
	D = 435
	box = [L, U, R, D]
	# Add box to image
	draw = ImageDraw.Draw(img)
	draw.rectangle(box)
	# Export image
	img.save('{}_out.jpg'.format(filename.split('.')[0]), 'JPEG')

box(sys.argv[1])
