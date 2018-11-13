# Protist Classifier
A neural network for protists image classification.

## Description:
This is a script that uses a real-time object detection convolutional neural network called YOLOv3 to classify different species of protists from a microscope image, or count the number of cells present in an image. Provided here are all the nessessary scripts to develop a database, train the YOLOv3 network, and perform a detection. Pre-trained weights are also available where we trained this neural network to count cells or to classify the following species within an image:

1. *Akashiwo sanguineae
2. Bysmatrum subsalsum
3. Durinskia baltica
4. Fukuyoa reutzleri
5. Gymnodinium aureolum
6. Gymnodinium impudicum
7. Heterocapsa sp.
8. Karlodinium arminger
9. Kryptoperidinium foliaceum
10. Levanderina fissa
11. Lingulodinium polyedrum
12. Margalefidinium polykrikoides
13. Pyrodinium bahamense
14. Scrippsiella sp
15. Scrippsiella trochoidea
16. Takayama tasmanica
17. Vulcanodinium sp.
18. Alexandrium affine
19. Alexandrium andersonii
20. Alexandrium fundyense
21. Alexandrium hiranoi
22. Alexandrium leei
23. Alexandrium monilatum
24. Alexandrium ostenfeldii
25. Coolia malayensis
26. Coolia monotis
27. Coolia palmyrensis
28. Coolia santacroce
29. Gambierdiscus belizeanus
30. Gambierdiscus caribaeus
31. Gambierdiscus caribeaus
32. Gambierdiscus carolinianus
33. Gambierdiscus carpenteri
34. Gambierdiscus pacificus
35. Karenia brevis
36. Karenia mikimotoi
37. Karenia papilionaceae
38. Karenia selliformis
39. Prorocentrum belizeanum
40. Prorocentrum cordatum
41. Prorocentrum elegans
42. Prorocentrum hoffmannianum
43. Prorocentrum lima
44. Prorocentrum micans
45. Prorocentrum rhathymum
46. Prorocentrum texanum
47. Prorocentrum triestinum
48. Amphidinium carterae
49. Amphidinium cf. thermaeum
50. Amphidinium cf. massartii
51. Amphidinium fijiensis
52. Amphidinium gibbossum
53. Amphidinium magnum
54. Amphidinium massartii
55. Amphidinium paucianulatum
56. Amphidinium pseudomassartii
57. Amphidinium theodori
58. Amphidinium thermaeum
59. Amphidinium tomasii
60. Amphidinium trulla*

## How to use:
This is a (Video)[] on how to use this setup.

### Update your system and install libraries.

This script works on GNU/Linux Ubuntu 18.04 and over using Python 3.6 and over. To use this script you will need to first update your system and install the dependencies using the following commands:

`sudo apt update`
`sudo apt full-upgrade`
`sudo apt install python3-pip`
`pip3 install numpy keras tensorflow PIL tkinter matplotlib imgaug scipy`

### Setting up a dataset
1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.
2. In the file named *class.txt* add the labels (classes) of each item you want to classify in a new line.
3. Open the GUI annotation tool using the following command: `python3 BBox.py`.
4. Click "Image Input Folder" on the top left to choose the directory that contains the images (./dataset/images).
5. Click "Label Output Folder" on the top left to choose the directory that will save the lables (./dataset/annotations).
6. Click "Load Dir" on the top right to load your choices (nothing will happen). Note: It is better to stick to the default dataset paths: ./dataset/images ./dataset/annotations ./dataset/check otherwise you will have to changes to different paths from within the code in some scripts. The image may not scale very well, make sure you see the entire image and not just part of it, change the values (currently at 1000) in line 273 of the BBox.py script accordindly (larger values = more zoomed image).
7. Use the mouse to generate a bounding box arround your object of interest.
8. Click "Next >>" on the bottom to load the first image.
9. Label images with boxes.
10. Click "Next >>" to save labels and move on to the next image.
11. Check to make sure that your annotations are correct by using the following command: `python3 txt-xml+check.py -cd ./dataset`.
12. The annotations are in text format and they need to be in XML format, to convert run the following command: `python3 txt-xml+check.py -t ./dataset/annotations ./dataset/images`.
13. If you are confident that your annotations are good choose `yes` to delete the text files and keep only the XML files.

### Training the neural network
For YOLOv2:
1. On line 48 of the YOLOv2.py file add all your labels in a list as such ["label 1", "label 2", "label 3"].
2. Download the YOLOv2 pre-trained weights using the following command: `wget https://pjreddie.com/media/files/yolov2.weights`.
3. Run training using the following command: `python3 YOLOv2.py -t`.
4. If the neural network training does not go well, you will have to change the network hyperparameters which are found on lines 49-63.
5. The logs directory contains the training logs. View the data using the following command: `tensorboard --logdir=./logs`.
6. The weights.h5 file is the weights file used for image detection.

For YOLOv3:
1. On line 5 of the config.json file add all your labels in a list as such ["label 1", "label 2", "label 3"].
2. Download the YOLOv3 pre-trained weights using the following command: `wget https://pjreddie.com/media/files/yolov3.weights`.
3. Download the RBC pre-trained weights using the following command: `wget '''wget https://onedrive.live.com/download.aspx?cid=5FDEBAB7450CDD92&authKey=!AO4VWYpzRLRXp3w&resid=5FDEBAB7450CDD92!136&ithint=.h5''' && mv *.h5 rbc.h5`
4. The network is resource heavy and required a GPU and 16GB RAM to run. Therefore ........................
5. Run training using the following command: `python3 YOLOv3.py -t`.
6. If the neural network training does not go well, you will have to change the network hyperparameters which are found in the config.jason file.
7. The logs directory contains the training logs. View the data using the following command: `tensorboard --logdir=./logs`.
8. The protist.h5 file is the weights file used for image detection.

### Detection
1. Run an image detection using the following command: `python3 YOLOv3.py -d FILENAME.jpg` or `python3 YOLOv2.py -d FILENAME.jpg`
