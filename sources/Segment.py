from PIL import Image

def segment(filename, size=(500, 500)):
	kernel = size
	stride = kernel
	img = Image.open(filename)
	pix = img.getdata()
	W, H = Image.open(filename).size
	L, U, R, D = 0, 0, kernel[0], kernel[1]
	count = 1
	for hight in range(int(H/kernel[1])):
		for width in range(int(W/kernel[0])):
			box = (L, U, R, D)
			region = img.crop(box)
			region.save('{}.jpg'.format(count), 'JPEG')
			count += 1
			L += stride[0]
			R += stride[0]
		L = 0
		U += stride[1]
		R = stride[0]
		D += stride[1]

segment('diatoms.jpg', size=(1000, 900))
