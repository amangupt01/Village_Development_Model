import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
from skimage import io, transform
import sys
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score

model_path = sys.argv[1]
IMAGE_DIR = sys.argv[2]
out_var = sys.argv[3]
extra = sys.argv[4]

model_path += out_var+extra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    #normalize the images with imagenet data mean and std
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#Load the output dataframe and save indices in a set
OUT = pickle.load(open('../Data/OUT.pickle','rb'))

#Load the train image list and the validation image list
test_list = pickle.load(open('../Data/val_list.pickle','rb'))

print("Test length:", len(test_list))

class IDataset:

	def __init__(self, img_dir, name_list, OUT):

		self.img_dir = img_dir
		self.name_list = name_list
		self.OUT = OUT

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		
		vill_id = float(self.name_list[idx].split('@')[2].split('.')[0])
		outval = self.OUT[out_var][vill_id]
			

		img_name = os.path.join(self.img_dir, self.name_list[idx])
		
		# try:
		image = io.imread(img_name)
		image = img_transform(image)

		sample = {'image': image, 'outval' : outval}
		return sample
		# except:
		# 	return {'image': torch.Tensor(np.zeros((3,150,150))), 'outval' : -1}

model_conv = torch.load(model_path,map_location = device)
model_conv.eval()

from torch.utils.data import Dataset, DataLoader
test_dataset = IDataset(IMAGE_DIR, test_list, OUT)
BATCH_SIZE = 50
NUM_WORKERS = 8
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

network = model_conv
count = 0

test_outs, test_preds = [], []

for sample in test_loader:
	count += 1
	images, outs = sample['image'], sample['outval']
	valids = np.where(outs != -1)[0]

	outs = outs[valids].float()
	images = images[valids]
	

	if torch.cuda.is_available():
		images = images.cuda()
		outs = outs.cuda()
	
	preds = network(images)
	preds = preds.view(-1)
	
	test_outs.extend(outs.tolist())
	test_preds.extend(preds.tolist())


	# print(outs)
	# print(preds)
	# print()
	# if count == 10:
		# break

print("Test R2 score: {}".format(r2_score(test_outs, test_preds)))