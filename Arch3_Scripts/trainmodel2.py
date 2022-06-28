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


#Where you want to save the model
model_path = sys.argv[1]

#Directory containing all the images in a state-wise fashion
IMAGE_DIR = sys.argv[2]
#The regression output you're training for ("FC_INT" in all caps, with the underscore in between, for example. Needs to be in this particular format.)
out_var = sys.argv[3]

#Freeze from the "freeze_layer"th last layer onwards (Expects freeze_layer in -ve, so the input should be -3 if you want to train from the 3rd last layer onwards and freeze the rest of the layers)
freeze_layer = int(sys.argv[4])

#If you want to add something extra to the model name.
model_name_add = sys.argv[5]

transfer_learning_model_path = sys.argv[6]

#Full model path
model_path += out_var+model_name_add
print("Model_path:",model_path)
print("Transfer Learning Model path:",transfer_learning_model_path)

#Tracks whether GPU is available. If so, then tensor variables can be loaded on to GPU, else on to CPU. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## You want to load the previous model and start training 
## or start the entire training
pretraining = False
transferLearning = True

# The required image transform before passing image as model input
img_transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    #normalize the images with imagenet data mean and std
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#Load the output dataframe
OUT = pickle.load(open('../Data/OUT.pickle','rb'))

#Load the train image list and the validation image list
train_list = pickle.load(open('../Data/train_list.pickle','rb'))
val_list = pickle.load(open('../Data/test_list.pickle','rb'))

print("Train length:", len(train_list))
print("Validation Length", len(val_list))

#The dataset to load the images, do preprocessing (passing through transform) and return them in tensor form to be passed as model input
class IDataset:

	def __init__(self, img_dir, name_list, OUT):

		self.img_dir = img_dir
		self.name_list = name_list
		self.OUT = OUT

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		
		#get the 2011 village id from the filename of the village image
		vill_id = float(self.name_list[idx].split('@')[2].split('.')[0])

		#Extract the corresponding value for the output (say FC_INT)
		outval = self.OUT[out_var][vill_id]/100
			

		img_name = os.path.join(self.img_dir, self.name_list[idx])
		
		# try:
		image = io.imread(img_name)
		image = img_transform(image)

		sample = {'image': image, 'outval' : outval}
		return sample
		# except:
		# 	return {'image': torch.Tensor(np.zeros((3,150,150))), 'outval' : -1}


if pretraining == True:
    model_conv = torch.load(model_path,map_location = device)
else: 
	#I made a neural network regression layer with 3 layers
	if transferLearning==True:
		model_conv = torch.load(transfer_learning_model_path,map_location = device)
		model_conv.fc = nn.Sequential(
        nn.Linear(in_features = 2048,out_features = 1,bias= True),
        nn.ReLU())
		if torch.cuda.is_available():
			print('Cuda is available:transferLearning')
			model_conv =  model_conv.cuda()
		else:
			print('Cuda is not available:transferLearning')	
	else:	
	    model_conv = torchvision.models.resnext50_32x4d(pretrained=True)
	    model_conv.fc = nn.Sequential(
	        nn.Linear(in_features = 2048,out_features = 512,bias= True),
	        nn.ReLU(),
	        nn.Linear(in_features = 512,out_features = 128,bias= True),
			nn.ReLU(),
			nn.Linear(in_features = 128,out_features = 1,bias= True)
	    )
	    #print(model_conv.parameters)
	    # model_conv.fc = nn.Linear(in_features = 2048,out_features = 1,bias= True)
	    if torch.cuda.is_available():
	        print('Cuda is available')
	        model_conv =  model_conv.cuda()


# model_conv.eval()

# Freeze certain layers in the network
for param in model_conv.parameters():
    param.requires_grad = False

for layer in list(model_conv.children())[freeze_layer:]:
	for param in layer.parameters():
		param.requires_grad = True


# Use MSE Loss for regression
criterion = nn.MSELoss()


# Create the train and validation datasets
from torch.utils.data import Dataset, DataLoader
train_dataset = IDataset(IMAGE_DIR, train_list, OUT)
val_dataset = IDataset(IMAGE_DIR, val_list, OUT)
BATCH_SIZE = 50
NUM_WORKERS = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)


# Do the training
num_epochs = 200

network = model_conv
optimizer = optim.Adam(network.parameters())

def save_model(model, path):
	torch.save(model, path)

if pretraining == False:
	prev_val_r2 = float('-inf')
else:
	prev_val_r2 = pickle.load(open('last_r2_score.pickle','rb'))

train_scores, test_scores = [], []

for epoch in range(num_epochs):
	avg_loss, batch = 0, 0
	count = 0
	train_outs, train_preds = [], []

	for sample in train_loader:
		batch += 1
		images, outs = sample['image'], sample['outval']
		valids = np.where(outs != -1)[0]
		if len(valids) == len(outs):
			count += 1

		outs = outs[valids].float()
		images = images[valids]
		

		if torch.cuda.is_available():
			images = images.cuda()
			outs = outs.cuda()
		
		preds = network(images)
		preds = preds.view(-1)
		loss = criterion(preds,outs)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		avg_loss += float(loss)

		# Save the actual outputs and predictions to calculate R2 score at the end of the epoch
		train_outs.extend(outs.tolist())
		train_preds.extend(preds.tolist())
		# break


	save_model(network, model_path)
	avg_loss /= batch
	print("Total Batches={} Valid Batches={}".format(batch, count))
	print("Average mean square error in epoch {} : {}".format(epoch+1,avg_loss))
	train_r2 = r2_score(train_outs, train_preds)
	print("Train R2 score in epoch {} : {}".format(epoch+1,train_r2))

	# Check out validation score every 3rd epoch and stop training if current validation R2 score is less than previous calculated (Early Stopping)
	if (epoch+1)%1 == 0:
		val_outs, val_preds = [], []

		for sample in val_loader:
			images, outs = sample['image'], sample['outval']
			valids = np.where(outs != -1)[0]

			outs = outs[valids].float()
			images = images[valids]
			
			if torch.cuda.is_available():
				images = images.cuda()
				outs = outs.cuda()
			
			preds = network(images)
			preds = preds.view(-1)
			
			# Save the actual outputs and predictions to calculate R2 score at the end
			val_outs.extend(outs.tolist())
			val_preds.extend(preds.tolist())
			# break


		curr_val_r2 = r2_score(val_outs, val_preds)
		print("Validation R2 score in epoch {} : {}".format(epoch+1, curr_val_r2))
		# if curr_val_r2 < prev_val_r2:
		# 	print("Early stopping reached!")
		# 	break
		# prev_val_r2 = curr_val_r2
		# pickle.dump(prev_val_r2, open('last_r2_score.pickle','wb'))

	train_scores += [train_r2]
	test_scores += [curr_val_r2]
	# pickle.dump(train_scores, open('Score_Track/train_scores_bfrud.pickle', 'wb'))
	# pickle.dump(test_scores, open('Score_Track/test_scores_bfrud.pickle', 'wb'))
	print()
	# break

print(train_scores)
print(test_scores)


	






