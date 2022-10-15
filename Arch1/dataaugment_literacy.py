#torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import Image

import sys
import numpy as np
import skimage.io as io
from skimage.io import imread, imsave, imshow
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage

import warnings
warnings.filterwarnings('ignore')
from skimage.transform import AffineTransform, warp

import os
import random
import pickle as pkl

directory = '/home/cse/mtech/mcs192554/scratch/literacy/train/cluster1'
extra_image = int(sys.argv[1])

augimg1 = []
for filename in os.listdir(directory):
	flag = 2
	if filename.endswith('.png'):
		flag = 0
		try :
			img = Image.open(os.path.join(directory,filename))
			img.verify()
		except (IOError, SyntaxError) as e:
			print('Bad File : ', filename)
			flag = 1
		
		if flag == 0 :
			image3 = imread(os.path.join(directory, filename))
			image3 = image3/255
			augimg1.append(image3)

random.shuffle(augimg1)

final_train = []
for i in tqdm(range(len(augimg1))):
	if(i<=extra_image):
		final_train.append(augimg1[i])
		final_train.append(rotate(augimg1[i], angle=random.randrange(360), mode = 'wrap'))

	else:
		final_train.append(augimg1[i])
#with open('test.pkl','wb') as f:
#	pkl.dump(final_train, f)

from skimage import img_as_ubyte

count = 1
#path = '/home/cse/mtech/mcs192554/scratch/msl_copy/train/finaltrain/train/cluster2/'
path = '/home/cse/mtech/mcs192554/scratch/literacy/train/cluster1_100percAug_made/'

for images in final_train:
    io.imsave(os.path.join(path,"imc%d.png"%count), img_as_ubyte(images))
    count = count + 1
