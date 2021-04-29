import pickle
import numpy as np

def dice(Gt='GT.pkl', Pr='PR.pkl', labels=[1, 1, 1, 1]):
	''' Calculates the Dice metric for polygon (pixel wise) detection '''
	with open(Gt, 'rb') as f: GT = pickle.load(f)
	with open(Pr, 'rb') as f: PR = pickle.load(f)
	for i in GT:
		Dice = []
		smooth = .001
		for a in range(len(labels)):
			gt = GT[i][:,:,a:a+1]
			for b in range(len(labels)):
				pr = PR[i][:,:,b:b+1]
				if 1. in pr and 1. in gt:
					intersection = np.sum(np.abs(pr * gt))
					mask_sum = np.sum(np.abs(gt)) + np.sum(np.abs(pr))
					union = mask_sum - intersection
					iou = (intersection + smooth) / (union + smooth)
					dice = 2*(intersection + smooth)/(mask_sum + smooth)
					gt_bg = GT[i][:,:,-1:]
					pr_bg = PR[i][:,:,-1:]
					intersection_bg = np.sum(np.abs(pr_bg * gt_bg))
					mask_sum_bg = np.sum(np.abs(gt_bg)) + np.sum(np.abs(pr_bg))
					union_bg = mask_sum_bg - intersection_bg
					iou_bg = (intersection_bg + smooth) / (union_bg + smooth)
					dice_bg = 2*(intersection_bg + smooth)/(mask_sum_bg + smooth)
					if iou > 0.5:
						avg = (dice+dice_bg)/2
						Dice.append(avg)
	mean_Dice = sum(Dice)/len(Dice)
	print(mean_Dice)
