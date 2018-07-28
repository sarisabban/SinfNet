from PIL import Image
#sudo pacman -S python-pillow

img = Image.open('earth.jpg')
pix = img.getdata()
# Image Dimentiona
x = 145
y = 145
# Kernel (30x30)px every 5px
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
		region.save('{}.jpeg'.format(count), 'JPEG')
		count += 1
		R += step
		L += step
	L = 0
	U += step
	R = 30
	D += step
