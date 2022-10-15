import torch

import sys

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision

from torch.autograd import Variable

from torchvision import datasets, models, transforms

import os

import numpy as np

from trainloop import RunManager,RunBuilder

from collections import OrderedDict

from tqdm import tqdm

from sys import argv





model_name = argv[1]

train_dir = argv[2]
num_e = argv[4]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## You want to load the previous model and start training

## or start the entire training

pretraining = False



data_transforms = {

    'train': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485,0.456,0.406] , [0.229,0.224,0.225])

    ]),

    'val': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485,0.456,0.406] , [0.229,0.224,0.225])

    ])

}





# Specify the data directory

data_dir = train_dir



## Provide the train and validation set for the training images

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}









if pretraining == True:

    model_conv = torch.load(model_name,map_location = device)

    model_conv.train()



else:

    model_conv = torchvision.models.resnext50_32x4d(pretrained=True)

    model_conv.fc = nn.Linear(in_features = 2048,out_features = 3,bias= True)

    #print(model_conv.parameters)

    if torch.cuda.is_available():

        print('Cuda is available')

        model_conv =  model_conv.cuda()





# Freeze all layers in the network or non freeze and train from scratch

for param in model_conv.parameters():

    param.requires_grad = True







weights_str = (sys.argv[3].split(','))
weights_list = []

for classwt in weights_str :
	weights_list.append(float(classwt))

weights = torch.tensor(weights_list, device=device)

criterion = nn.CrossEntropyLoss(weight=weights)



params = OrderedDict(

    batch_size = [128],

    shuffle = [True]

)





## Set the number of epochs

num_epochs = int(num_e)





m = RunManager()



for run in RunBuilder.get_runs(params):



    network = model_conv

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x] , batch_size = run.batch_size,

                                                    shuffle = run.shuffle,num_workers = 8) for x in ['train','val']}



    optimizer = optim.Adam(network.parameters())





    m.begin_run(run,network,dataloaders['train'])



    for epoch in range(num_epochs):

        m.begin_epoch()



        batch = 0

        for images,labels in tqdm(dataloaders['train']):

            batch += 1

            #print(batch)

            images = Variable(images)

            labels = Variable(labels)



            if torch.cuda.is_available():

                images = images.cuda()

                labels = labels.cuda()



            preds = network(images)

            loss = criterion(preds,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            ## Send the loss and preds labels to manager

            m.track_loss(loss)

            m.track_num_correct(preds,labels)

            #m.save_results('results_resnet')
            m.save_model(model_name)





        m.end_epoch()

    m.end_run()
