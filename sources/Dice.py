import pickle
import numpy as np

def dice(Gt='GT.pkl', Pr='PR.pkl'):
	''' Calculates the Dice metric for polygon (pixel wise) detection '''
	with open(Gt, 'rb') as f: GT = pickle.load(f)
	with open(Pr, 'rb') as f: PR = pickle.load(f)
	Dice = []
	for i in GT:
		gt = GT[i][:,:,-1]
		pr = PR[i][:,:,-1]
		smooth = .001
		intersection = np.sum(np.abs(pr * gt))
		mask_sum = np.sum(np.abs(gt)) + np.sum(np.abs(pr))
		union = mask_sum - intersection
		iou = (intersection + smooth) / (union + smooth)
		dice = 2*(intersection + smooth)/(mask_sum + smooth)
		Dice.append(dice)
	mean_Dice = sum(Dice)/len(Dice)
	print(mean_Dice)
