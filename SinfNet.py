#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from sources import mAP
from sources import CNN
from sources import YOLOv3
from sources import Augment
from sources import Semantic
from sources import Translate
from sources import Miscellaneous

parser = argparse.ArgumentParser(description='Collection of datasets and networks for microscope organism classification')
parser.add_argument('-ac',  '--augment_cnn',      nargs='+', help='Augment whole images for a CNN')
parser.add_argument('-ap',  '--augment_poly',     nargs='+', help='Augment images with bounding polygons')
parser.add_argument('-ab',  '--augment_bbox',     nargs='+', help='Augment images with bounding boxes')
parser.add_argument('-B' ,  '--biomass',          nargs='+', help='Calculate biomass')
parser.add_argument('-bb',  '--bbox_results',     nargs='+', help='Plot bounding box results')
parser.add_argument('-C' ,  '--crop',             nargs='+', help='Make image dimentions multiples of 32')
parser.add_argument('-Cb',  '--check_box',        nargs='+', help='Check bounding box annotation after augmentation')
parser.add_argument('-Cp',  '--check_poly',       nargs='+', help='Check bounding polygon annotation after augmentation')
parser.add_argument('-Cob', '--csv_coco',         nargs='+', help='Convert .csv to coco .json for bounding boxes')
parser.add_argument('-ct',  '--cnn_train',        nargs='+', help='Train on CNN')
parser.add_argument('-cp',  '--cnn_predict',      nargs='+', help='Predict from CNN')
parser.add_argument('-mAP', '--mAP_calc',         nargs='+', help='Calculate the mean average precision of bounding box results')
parser.add_argument('-S' ,  '--segment',          nargs='+', help='Segment a large image')
parser.add_argument('-st',  '--semantic_train',   nargs='+', help='Train on UNet')
parser.add_argument('-sp',  '--semantic_predict', nargs='+', help='Predict from UNet')
parser.add_argument('-tb',  '--translate_bbox',   nargs='+', help='Translate between annotation file formats for bounding boxes')
parser.add_argument('-tp',  '--translate_poly',   nargs='+', help='Translate between annotation file formats for bounding polygons')
parser.add_argument('-v' ,  '--via',              action='store_true', help='Open image labeling tool')
parser.add_argument('-ot',  '--object_train',     nargs='+', help='Train on YOLOv3')
parser.add_argument('-op',  '--object_predict',   nargs='+', help='Predict from YOLOv3')
args = parser.parse_args()

def main():
	if args.augment_cnn:
		Augment.augment_cnn(	input_path=sys.argv[2],
								output_path=sys.argv[3],
								count=int(sys.argv[4]))
	elif args.augment_bbox:
		Augment.augment_bbox(	image_input=sys.argv[2],
								image_output=sys.argv[3],
								bbox_input=sys.argv[4],
								bbox_output=sys.argv[5],
								input_format=sys.argv[6],
								output_format=sys.argv[7],
								count=int(sys.argv[8]))
	elif args.augment_poly:
		Translate.translate_poly(image_path=sys.argv[2],
								ann_input=sys.argv[3],
								ann_output=sys.argv[4],
								input_format=sys.argv[5],
								output_format=sys.argv[6])
		for filename in os.listdir(sys.argv[2]):
			name = filename[:-4]
			image = '{}/{}.jpg'.format(sys.argv[2], name)
			annot = '{}/{}.json'.format(sys.argv[4], name)
			Augment.augment_poly(image_input=image,
								image_output=sys.argv[7],
								poly_input=annot,
								poly_output=sys.argv[8],
								count=sys.argv[9])
	elif args.biomass:
		Width_img = int(sys.argv[2])
		Hight_img = int(sys.argv[3])
		Depth_img = int(sys.argv[4])
		width     = int(sys.argv[5])
		hight     = int(sys.argv[6])
		whitePX   = int(sys.argv[7])
		Miscellaneous.Biomass(	Width_img,
								Hight_img,
								Depth_img,
								width,
								hight,
								whitePX)
	elif args.bbox_results:
		directory = sys.argv[2]
		gt = sys.argv[3]
		pr = sys.argv[4]
		for image in os.listdir(directory):
			Miscellaneous.plot_bbox_results('{}/{}'.format(directory, image), gt=gt, pr=pr)
	elif args.check_box:
		images = sys.argv[2]
		annots = sys.argv[3]
		for i in os.listdir(images):
			I = images
			A = annots
			f = i.split('.')[0]
			Miscellaneous.confirm_box(
								'{}/{}.jpg'.format(I, f),
								'{}/{}.xml'.format(A, f))
	elif args.check_poly:
		images = sys.argv[2]
		annots = sys.argv[3]
		for i in os.listdir(images):
			I = images
			A = annots
			f = i.split('.')[0]
			Miscellaneous.confirm_poly(
								'{}/{}.jpg'.format(I, f),
								'{}/{}.json'.format(A, f))
	elif args.csv_coco:
		Miscellaneous.csv_to_coco(
								img_dir=sys.argv[2],
								gt=sys.argv[3],
								pr=sys.argv[4])
	elif args.crop:
		Miscellaneous.crop(filename=sys.argv[2])
	elif args.cnn_train:
		CNN.CNN(network=sys.argv[2],
				choice='train',
				weights='',
				Train=sys.argv[3],
				Valid=sys.argv[4],
				Tests=sys.argv[5],
				prediction='')
	elif args.cnn_predict:
		CNN.CNN(network=sys.argv[2],
				choice='predict',
				weights=sys.argv[3],
				Train='',
				Valid='',
				Tests='',
				prediction=sys.argv[4])
	elif args.mAP_calc:
		mAP.mAP(sys.argv[2], sys.argv[3])
	elif args.segment:
		Miscellaneous.segment(	filename=sys.argv[2],
								size=(int(sys.argv[3]), int(sys.argv[4])))
	elif args.semantic_train:
		Semantic.train()
	elif args.semantic_predict:
		Semantic.predict(sys.argv[5])
	elif args.translate_bbox:
		Translate.translate_bbox(image_path=sys.argv[2],
								ann_input=sys.argv[3],
								ann_output=sys.argv[4],
								input_format=sys.argv[5],
								output_format=sys.argv[6])
	elif args.translate_poly:
		Translate.translate_poly(image_path=sys.argv[2],
								ann_input=sys.argv[3],
								ann_output=sys.argv[4],
								input_format=sys.argv[5],
								output_format=sys.argv[6])
	elif args.via:
		os.system('firefox ./sources/VIA.html')
	elif args.object_train:
		YOLOv3.train()
	elif args.object_predict:
		YOLOv3.predict(sys.argv[2], sys.argv[4], './')

if __name__ == '__main__': main()
