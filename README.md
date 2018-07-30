# Plankton Classifier
A neural network plankton image classifier.

## Requirements:
This script work on GNU/Linux. To use this script you will need to update your system and install the following libraries:

`sudo apt update && sudo apt full-upgrade && sudo apt install python3-pip python3-numpy python3-tensorflow && pip3 install keras`

## Description:
This is a script that uses a convolutional neural network to classify different species of plankton by simply providing a microscope image of the cell of interest.

## How To Use:
Update your system and install the required libraries.

### Setting up a dataset and training the neural network
The neural network is already trained, but if you want to develop your own dataset or update the one provided you can use the following commands to setup a new dataset and train the neural network on it (the dataset is used uncompressed):

1. Take a picture of your sample under a bright field light microscope at magnification ...x. The image must be coloured, ...x... pixels at resolution .....  and in JPEG format and should contain many cells.
2. Segment the image into smaller images using the following command:

`python3 ....py segment IMAGE.jpg SPECIES_NAME`

IMAGE: is the image from the microscope.
SPECIES_NAME: the name of your species.

This will result in ..... number of images from the original microscope image placed in a directory named after the species name you defined. Some images will have cells, others will not. Thus use the following command to separate images that contain cells from those that are empty:

`python3 ....py sort DIRECTORY`

DIRECTORY: is the directory that contains your images to be sorted (should be the species name you defined).

3. Augment the images to increase your dataset using the following command:

`python3 ....py augment DIRECTORY`

DIRECTORY: is the directory that contains your images to be augmented (should be the species name you defined).

This takes each image and changes it a little bit (flips, mirrors etc...). This is used to increase our dataset significantly.

4. Repeat this process ....x for each species to collect enough images (try to get arround ..... images for each species).

5. Now it is time to put together the dataset. Use the following command to setup the dataset:

`python3 ....py dataset DIRECTORY1 DIRECTORY2 DIRECTORY3 DIRECTORY4 DIRECTORY5`

DIRECTORY1-5: are the directories (named after each species) that contains each species images.

The dataset will be a directory called *dataset*. This dataset directory will be setup using the conventional way: within the directory there is are *tests* *train* *valid* directories that each contains a directory of each species's images:

dataset
	|
	-tests
		|
		-species1
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species2
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species3
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
	-train
		|
		-species1
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species2
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species3
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
	-valid
		|
		-species1
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species2
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg
		-species3
			|
			-image1.jpg
			-image2.jpg
			-image3.jpg

6. Train the neural network on the dataset using the following command:

`python3 ....py train DATASET`

DATASET: the directory that is called dataset that contains your dataset.

Training time is around ..... hours, and the script will output a weights file called weight.h5 (this is the result of the training process and needs to be included in the same directory as this script when asked to identify a new image).

### Identify plankton
This script comes with a neural network that is already trained to identify the following species:

1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 

Thus, if you want to use the current model, or have trained your own model, simply use the following command:

`python3 ....py identify IMAGE.jpg`

The image must be ...x... pixels, in colour, and in JPEG format.

Make sure the weights.h5 file is included in the same directory as this script (this file containes the results of the neural network training process).
