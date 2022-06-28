import os
import sys
import pickle
import numpy as np
from numpy import random as rd

IMAGE_DIR = sys.argv[1]
OUT = pickle.load(open('../Data/OUT.pickle','rb'))
val_ids = set(list(OUT.index))

#Filter out only the required images
final_list = []
for st in os.listdir(IMAGE_DIR):
	for im in os.listdir(os.path.join(IMAGE_DIR,st)):
		vill_id = float(im.split('@')[2].split('.')[0])
		if vill_id in val_ids:
			final_list += [os.path.join(st,im)]


length = len(final_list)
print("Total length:", length)

rand_ids = rd.permutation(length)
print(rand_ids)
final_list = np.array(final_list)

train_perc, val_perc = 0.7, 0.2
divide1, divide2 = int(train_perc*length), int((train_perc+val_perc)*length)
train_list, val_list, test_list = final_list[rand_ids[:divide1]], final_list[rand_ids[divide1:divide2]], final_list[rand_ids[divide2:]]

print("Length of training data:", len(train_list))
print("Length of validation data:", len(val_list))
print("Length of testing data:", len(test_list))

pickle.dump(list(train_list), open('../Data/train_list.pickle', 'wb'))
pickle.dump(list(val_list), open('../Data/val_list.pickle', 'wb'))
pickle.dump(list(test_list), open('../Data/test_list.pickle', 'wb'))



