# Protist Image Classifier
A neural network for protists image classification.

## Description:
This is a script that uses a real-time object detection convolutional neural network called YOLOv3 to detection cells present in an image, or classify different species of protists from a microscope image. Provided here are all the necessary scripts to develop a database, train the **YOLOv3** networks, and perform a detection. Microscope image datasets and pre-trained weights are also available where we trained this neural network to detect cells or to classify the following species within an image:

<p align="center">Protist dataset - 60 species:</p>

<sub>*Akashiwo sanguineae, Bysmatrum subsalsum, Durinskia baltica, Fukuyoa reutzleri, Gymnodinium aureolum, Gymnodinium impudicum, Heterocapsa sp., Karlodinium arminger, Kryptoperidinium foliaceum, Levanderina fissa, Lingulodinium polyedrum, Margalefidinium polykrikoides, Pyrodinium bahamense, Scrippsiella sp, Scrippsiella trochoidea, Takayama tasmanica, Vulcanodinium sp., Alexandrium affine, Alexandrium andersonii, Alexandrium fundyense, Alexandrium hiranoi, Alexandrium leei, Alexandrium monilatum, Alexandrium ostenfeldii, Coolia malayensis, Coolia monotis, Coolia palmyrensis, Coolia santacroce, Gambierdiscus belizeanus, Gambierdiscus caribaeus, Gambierdiscus caribeaus, Gambierdiscus carolinianus, Gambierdiscus carpenteri, Gambierdiscus pacificus, Karenia brevis, Karenia mikimotoi, Karenia papilionaceae, Karenia selliformis, Prorocentrum belizeanum, Prorocentrum cordatum, Prorocentrum elegans, Prorocentrum hoffmannianum, Prorocentrum lima, Prorocentrum micans, Prorocentrum rhathymum, Prorocentrum texanum, Prorocentrum triestinum, Amphidinium carterae, Amphidinium cf. thermaeum, Amphidinium cf. massartii, Amphidinium fijiensis, Amphidinium gibbossum, Amphidinium magnum, Amphidinium massartii, Amphidinium paucianulatum, Amphidinium pseudomassartii, Amphidinium theodori, Amphidinium thermaeum, Amphidinium tomasii, Amphidinium trulla*</sub>

The cell detection setup it mainly used to construct the protist dataset, since each protist species require at least 1000 annotated images, the cell detection neural network is use to auto annotate microscope cell images and construct a dataset, that is why it is provided here.

## Available datasets and trained weight files
You can download these datasets to add to them and re-train the network to develop it, or to simply replicate our work:

| Dataset name                                                                                           | Weights                                                                  |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
|[Amoeba Active/Inactive Dataset](https://www.dropbox.com/s/a7z43eivfrzd0rx/Amoeba%28950%29.tar.bz2?dl=0)|[YOLOv3 Weights](https://www.dropbox.com/s/x044cdo7kznoeuf/Amoeba.h5?dl=0)|
|[Cell detection Dataset]()                                                                              |[YOLOv3 Weights]()                                                        |
|[Dinoflagellates Dataset]()                                                                             |[YOLOv3 Weights]()                                                        |

## How to use:
This is a [Video]() on how to use this setup.

### Update your system and install libraries.
This script works on GNU/Linux Ubuntu 18.04 and over using Python 3.6 and over. To use this script you will need to first update your system and install the dependencies using the following commands:

`sudo apt update`

`sudo apt full-upgrade`

`sudo apt install python3-pip`

`pip3 install numpy keras tensorflow opencv-python tkinter matplotlib imgaug scipy`

### Setting up a dataset
If you want to develop your own dataset follow these steps:

1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.
2. In the Label.py script add the labels (classes) of each item the list in line 64 and save the file.
3. Make a directory called dataset and within it in make the following directories: Images, BBox_Annotations, Annotations, and Check. You should have the following structure:

*./dataset/Annotations*

*./dataset/Images*

*./dataset/Test*

*./dataset/BBox_Annotations*

*./dataset/Check*

Your images should be in *./dataset/Images* of course. It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script. So just stick to these to keep it simple.

4. Open the GUI annotation tool using the following command:

`python3 Label.py --bbox`

5. Click "Image Input Folder" on the top left to choose the directory that contains the images (./dataset/Images).
6. Click "Label Output Folder" on the top left to choose the directory that will save the labels (./dataset/BBox_Annotations).
7. Click "Load Dir" on the top right to load your choices (nothing will happen). Note: It is better to stick to the default dataset paths mentioned in step 3, otherwise you will have to changes to different paths from within the code in some scripts. The images may not scale very well, make sure you see the entire image and not just part of it, change the values (currently at 700) in line 280 of the BBox.py script accordingly (larger values = more zoomed image).
8. You must click "Next" to load the images (but it will skip the first image, so go back to it).
9. Use the mouse to generate a bounding box around each object of interest.
10. Label each box with the labels from the drop down menu on the right and clicking "Confirm Class".
11. Click "Next >>" to save the labels and move on to the next image (images are not loaded by filename order).
12. Once finished, check to make sure that your annotations are correct by using the following command:

`python3 Label.py --check`

This will generate a new directory called ./dataset/Check with the images showing their annotations.

13. The annotations are in text (.txt) file format and they need to be in XML format, to convert run the following command:

`python3 Label.py --translate`

This will generate a new directory called ./dataset/Annotations and this directory will be used in the neural network.

14. If you want to rename a label throughout the dataset use the following command

`python3 Label.py --rename OLD NEW`

For help use this command:

`python3 Label.py --help`

15. Add some images to the Test direcotry which will be used to test the accuracy of the final trained network (on images the network has never seen).

### Training the neural network
1. On line 53 of the YOLOv3.py script add all your labels in a list as such ["label 1", "label 2", "label 3"], and on lines 57 and 58 change your output file names.
2. The network is resource heavy and required a large GPU and more than 16GB of RAM to run. Therefore some cloud GPU cloud services may not work and a larger system is required.
3. To see the help menu use the following command:

`python3 YOLOv3.py -h`

4. Run training using the following command:

`python3 YOLOv3.py -t`

5. If the neural network training does not go well, you will have to change the network hyper parameters which are found in lines 49-79 of the script file.
6. The logs directory contains the training logs. View the data using the following command:

`tensorboard --logdir=./logs`

7. The .h5 file is the weights file used for image detection.

### Detection
If you just want to run a detection without developing a dataset nor re-training the network you can just run this command right now using the weights of our trained network.
1. Download the relevant weights file, links available above.
2. Run image detection using the following command:

`python3 YOLOv3.py -d WEIGHTS.h5 FILENAME`

The FILENAME in YOLOv3.py can be either a .jpg image, .mp4 video or a webcam.

**You can add to an existing dataset**. Since manual annotations is time consuming, this same neural network can be used to annotate new images to expand the current dataset (instead of annotating 1000s of images manually), make sure you use the Cell.h5 weights since you want to detect only the cells in the images. Thus you must insure your image is made up of a pure strain of cells. Start by commenting in line 555 in the YOLOv3.py script and use the following command to loop through all images in a directory and annotate them.:

`for i in IMAGE_DIRECTORY/*; do f="${i##*/}"; python3 YOLOv3.py -d Cell.h5 $i > ./"${f%.*}".txt; rm ./"${i##*/}"; sed -i "s/[^ ]*$/Cell/" ./"${f%.*}".txt;cat ./"${f%.*}".txt | wc -l > temp && cat ./"${f%.*}".txt >> temp && mv temp ./"${f%.*}".txt; done`

I know the command is ugly, but it works. The only thing you have to change is the *IMAGE_DIRECTORY* at the start of the command. The annotation is as good as the training of the network, which is not 100%, therefore a human must go over the annotated images using the Label.py script as in step 4 to fix any mistakes. Make sure you repeat steps 12 and 13 to check and translate the new annotations. Annotations may have some mistakes therefore checking the annotations is very important.

**Contributing to our dataset**
If you would like to add images to our dataset (any type of protist cell) make sure that each species has 2000 annotated images where each image is sharp and taken from a brightfield light miscroscope at 400x magnification. Please contact me so we can work together.

## References:
When using these scripts kindly reference the following:
* 
