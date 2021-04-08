import numpy as np
from collections import defaultdict, Counter

class mAP():
	def __init__(self, Gt='Object.csv', Pr='Object_results.csv', iou_thr=0.5):
		self.gt = Gt
		self.pr = Pr
		self.iou_thr = iou_thr
		self.TPFP, self.PRlabels, self.GTBB = self.TP_FP()
		self.pre_rec = self.precision_recall(self.TPFP,self.PRlabels,self.GTBB)
		self.AP = self.AP_calc(self.pre_rec)
		self.mAP = self.mAP_calc(self.AP)
		print(self.mAP)
	def IOU(self, GTbox, PRbox):
		''' Calculates IOU '''
		GTx, GTy, GTw, GTh, GTL = GTbox
		PRx, PRy, PRw, PRh, PRL = PRbox
		GTx, GTy, GTw, GTh = int(GTx), int(GTy), int(GTw), int(GTh)
		PRx, PRy, PRw, PRh = int(PRx), int(PRy), int(PRw), int(PRh)
		if(GTw < PRx): return 0.0
		if(GTh < PRy): return 0.0
		if(GTx > PRw): return 0.0
		if(GTy > PRh): return 0.0
		GTbox_area = (GTw - GTx + 1) * (GTh - GTy + 1)
		PRbox_area = (PRw - PRx + 1) * (PRh - PRy + 1)
		X = np.max([GTx, PRx])
		Y = np.max([GTy, PRy])
		W = np.min([GTw, PRw])
		H = np.min([GTh, PRh])
		intersection_area = (W - X + 1) * (H - Y + 1)
		union_area = (GTbox_area + PRbox_area - intersection_area)
		IOU = intersection_area/union_area
		return(IOU)
	def TP_FP(self):
		''' Calculates true positives and false positives of each label '''
		GT = defaultdict(list)
		with open(self.gt) as f:
			next(f)
			for line in f:
				line = line.strip().split(',')
				filename = line[0]
				size = int(line[1])
				confidence = line[2]
				x = int(line[6].split(':')[-1])
				y = int(line[7].split(':')[-1])
				w = int(line[8].split(':')[-1])
				h = int(line[9].split(':')[-1][:-2])
				label = line[10].split('"')[3]
				GT[filename].append((size, confidence, x, y, x+w, y+h, label))
		PR = defaultdict(list)
		with open(self.pr) as f:
			next(f)
			for line in f:
				line = line.strip().split(',')
				filename = line[0]
				size = int(line[1])
				confidence = float(line[2])
				x = int(line[6].split(':')[-1])
				y = int(line[7].split(':')[-1])
				w = int(line[8].split(':')[-1])
				h = int(line[9].split(':')[-1][:-2])
				label = line[10].split('"')[3]
				PR[filename].append((size, confidence, x, y, w, h, label))
		images          = list(GT.keys())
		Total_PR_Labels = dict(Counter([x[6] for y in PR.values() for x in y]))
		Total_GT_Boxes  = [x[6] for y in GT.values() for x in y]
		Total_PR_Boxes  = [x[6] for y in PR.values() for x in y]
		True_False = []
		for image in images:
			GT_idx=[]
			PR_idx=[]
			ious=[]
			for ipb, PRbox in enumerate(PR[image]):
				for igb, GTbox  in enumerate(GT[image]):
					iou = self.IOU(GTbox[2:], PRbox[2:])
					if iou > self.iou_thr and GTbox[6]==PRbox[6]:
						GT_idx.append(igb)
						PR_idx.append(ipb)
						ious.append(iou)
			iou_sort = np.argsort(ious)[::1]
			GT_total_idx = [x for x in range(len(GT[image]))]
			PR_total_idx = [x for x in range(len(PR[image]))]
			GT_match_idx = []
			PR_match_idx = []
			for idx in iou_sort:
				gt_idx = GT_idx[idx]
				pr_idx = PR_idx[idx]
				if(gt_idx not in GT_match_idx) and (pr_idx not in PR_match_idx):
					GT_match_idx.append(gt_idx)
					PR_match_idx.append(pr_idx)
			TP_idx  = PR_match_idx
			FN_idx  = list(set(PR_match_idx) - set(GT_match_idx))
			FP_temp = list(set(PR_total_idx) - set(GT_match_idx))
			FP_idx  = list(set(FP_temp) - set(FN_idx))
			for tpr, tpg in zip(TP_idx, GT_match_idx):
				True_False.append((PR[image][tpr][1], PR[image][tpr][6], 'TP'))
			for fpr in FP_idx:
				True_False.append((PR[image][fpr][1], PR[image][fpr][6], 'FP'))
		True_False.sort(reverse=True)
		return(True_False, Total_PR_Labels, len(Total_GT_Boxes))
	def precision_recall(self, True_False, Total_PR_Labels, Total_GT_Boxes):
		''' Calculates the precision and recall of each label'''
		precision_recall_per_label = {}
		for label in Total_PR_Labels:
			precision = []
			recall = []
			r_cd = 0
			p_cd = 0
			p_cn = 0
			for item in True_False:
				if item[1]==label:
					if item[2]=='TP':
						r_cd += 1
						p_cd += 1
						p_cn += 1
						precision.append(p_cn / p_cd)
						recall.append(r_cd / Total_PR_Labels[label])
					if item[2]=='FP':
						p_cd += 1
						precision.append(p_cn / p_cd)
						recall.append(r_cd / Total_PR_Labels[label])
			precision_recall_per_label[label] = (precision, recall)
		return(precision_recall_per_label)
	def AP_calc(self, precision_recall_per_label):
		''' Calculates area under curve to get each label's average precision '''
		average_precision = {}
		for label in precision_recall_per_label:
			P = precision_recall_per_label[label][0]
			R = precision_recall_per_label[label][1]
			area = np.trapz(P, R)
			average_precision[label] = area
		return(average_precision)
	def mAP_calc(self, average_precision):
		''' Calculates the mean average precision '''
		Ps = [x for x in average_precision.values()]
		mean_average_precision = sum(Ps)/len(Ps)
		return(mean_average_precision)
