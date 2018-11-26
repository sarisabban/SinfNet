# Protist Image Classifier
A neural network for protists image classification.

## Description:
This is a script that uses a real-time object detection convolutional neural network called YOLOv3 to classify different species of protists from a microscope image, or count the number of cells present in an image. Provided here are all the nessessary scripts to develop a database, train the YOLOv3 network, and perform a detection. Pre-trained weights are also available where we trained this neural network to count cells or to classify the following species within an image:

1. *Akashiwo sanguineae*
2. *Bysmatrum subsalsum*
3. *Durinskia baltica*
4. *Fukuyoa reutzleri*
5. *Gymnodinium aureolum*
6. *Gymnodinium impudicum*
7. *Heterocapsa sp.*
8. *Karlodinium arminger*
9. *Kryptoperidinium foliaceum*
10. *Levanderina fissa*
11. *Lingulodinium polyedrum*
12. *Margalefidinium polykrikoides*
13. *Pyrodinium bahamense*
14. *Scrippsiella sp*
15. *Scrippsiella trochoidea*
16. *Takayama tasmanica*
17. *Vulcanodinium sp.*
18. *Alexandrium affine*
19. *Alexandrium andersonii*
20. *Alexandrium fundyense*
21. *Alexandrium hiranoi*
22. *Alexandrium leei*
23. *Alexandrium monilatum*
24. *Alexandrium ostenfeldii*
25. *Coolia malayensis*
26. *Coolia monotis*
27. *Coolia palmyrensis*
28. *Coolia santacroce*
29. *Gambierdiscus belizeanus*
30. *Gambierdiscus caribaeus*
31. *Gambierdiscus caribeaus*
32. *Gambierdiscus carolinianus*
33. *Gambierdiscus carpenteri*
34. *Gambierdiscus pacificus*
35. *Karenia brevis*
36. *Karenia mikimotoi*
37. *Karenia papilionaceae*
38. *Karenia selliformis*
39. *Prorocentrum belizeanum*
40. *Prorocentrum cordatum*
41. *Prorocentrum elegans*
42. *Prorocentrum hoffmannianum*
43. *Prorocentrum lima*
44. *Prorocentrum micans*
45. *Prorocentrum rhathymum*
46. *Prorocentrum texanum*
47. *Prorocentrum triestinum*
48. *Amphidinium carterae*
49. *Amphidinium cf. thermaeum*
50. *Amphidinium cf. massartii*
51. *Amphidinium fijiensis*
52. *Amphidinium gibbossum*
53. *Amphidinium magnum*
54. *Amphidinium massartii*
55. *Amphidinium paucianulatum*
56. *Amphidinium pseudomassartii*
57. *Amphidinium theodori*
58. *Amphidinium thermaeum*
59. *Amphidinium tomasii*
60. *Amphidinium trulla*

## How to use:
This is a [Video]() on how to use this setup.

### Update your system and install libraries.
This script works on GNU/Linux Ubuntu 18.04 and over using Python 3.6 and over. To use this script you will need to first update your system and install the dependencies using the following commands:

`sudo apt update`

`sudo apt full-upgrade`

`sudo apt install python3-pip`

`pip3 install numpy keras tensorflow PIL opencv-python tkinter matplotlib imgaug scipy`

### Setting up a dataset
You can download here the [Cell detection Dataset](https://www.dropbox.com/s/3qm7xi12bbxgje7/dataset.tar.bz2?dl=0), the [Dinoflagellates Dataset](), or the full [Protist Dataset]() if you want to retrain the network or add to the dataset and train the network.

If you want to develop your own dataset follow these steps:

1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.
2. In the file named *class.txt* add the labels (classes) of each item that you want to classify, each label in a new line.
3. Make a directory called dataset and within it in make the following directories: Images, BBox_Annotations, Annotations, and Check. You should have the following structure:

*./dataset/Images*
*./dataset/BBox_Annotations*
*./dataset/Annotations*
*./dataset/Check*

It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script. So just stick to these to keep it simple.

4. Open the GUI annotation tool using the following command:

`python3 BBox.py`

5. Click "Image Input Folder" on the top left to choose the directory that contains the images (./dataset/Images).
6. Click "Label Output Folder" on the top left to choose the directory that will save the lables (./dataset/BBox_Annotations).
7. Click "Load Dir" on the top right to load your choices (nothing will happen). Note: It is better to stick to the default dataset paths mentioned in step 3, otherwise you will have to changes to different paths from within the code in some scripts. The images may not scale very well, make sure you see the entire image and not just part of it, change the values (currently at 700) in line 273 of the BBox.py script accordindly (larger values = more zoomed image).
8. You must click "Next" to load the images (but it will skip the first image, so go back to it).
9. Use the mouse to generate a bounding box arround each object of interest.
10. Label each box with the labels from the drop down menu on the right.
11. Click "Next >>" to save the labels and move on to the next image (images are not loaded by filename order).
12. Once finished, check to make sure that your annotations are correct by using the following command:

`python3 txt-xml+check.py -cd`

This will generate a new directory called ./dataset/Check with the images showing their annotations.

13. The annotations are in text (.txt) file format and they need to be in XML format, to convert run the following command:

`python3 txt-xml+check.py -t`

This will generate a new directory called ./dataset/Annotations and this directory will be used in the neural network.

### Training the neural network
#### For YOLOv2:
1. On line 46-71 of the YOLOv2.py script add all your labels in a list as such ["label 1", "label 2", "label 3"] and adjust the location of the dataset and the name of your output weights file.
2. Run training using the following command:

`python3 YOLOv2.py -t`

3. If the neural network training does not go well, you will have to change the network hyperparameters which are found on lines 46-71.
4. The logs directory contains the training logs. View the data using the following command:

`tensorboard --logdir=./logs`

5. The .h5 file is the weights file used for image detection.

#### For YOLOv3:
1. On line 5 of the config.json file add all your labels in a list as such ["label 1", "label 2", "label 3"].
2. The network is resource heavy and required a large GPU and more than 16GB of RAM to run. Therefore some cloud GPU cloud services may not work and a larger system is required.
3. Run training using the following command:

`python3 YOLOv3.py -t`

4. If the neural network training does not go well, you will have to change the network hyperparameters which are found in the config.jason file.
5. The logs directory contains the training logs. View the data using the following command:

`tensorboard --logdir=./logs`

6. The .h5 file is the weights file used for image detection.

### Detection
If you just want to run a detection without developing a dataset nor re-training the network you can just run this command right now using the weights of our trained network.
1. [Download](https://www.dropbox.com/sh/h6tjfbh3wymxze1/AABN1FslPRjgCnF-5S2i5jEpa?dl=0) the relevent weights file.
2. If you are going to use YOLOv3 then modify the config.json file to point to the weights file. If you are going to use YOLOv2 then just use the command (no need to modify the config.json file).

3. Run image detection using the following command:

`python3 YOLOv3.py -d FILENAME.jpg`

or

`python3 YOLOv2.py -d WEIGHTS.h5 IMAGE.jpg`

### Jupyter notebooks
The Jupyter notebooks are provided for YOLOv2 and YOLOv3 to be able to quickly implemnet these scripts on cloud GPUs.
