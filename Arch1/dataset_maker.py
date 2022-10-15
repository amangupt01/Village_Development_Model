import os
import pickle
from shutil import copy
from sys import argv

mainfilepath = argv[1] # path of folder where all processed images are
picklename = argv[2] # clusters are sent as pickle files
dirname = argv[3] # output directory name

save_path = {'train': [dirname + '/train/cluster1/',dirname + '/train/cluster2/',
                      dirname + '/train/cluster3/'],
              'val' :[dirname + '/val/cluster1/',dirname + '/val/cluster2/',
                      dirname + '/val/cluster3/'] }

images_path = mainfilepath

with open(picklename, "rb") as input_file:
    villages_list = pickle.load(input_file)



villages_list = villages_list.values

set_advance = set()
set_rudimentary = set()
set_intermediate = set()


#for asset this is the line : set_rudimentary.add(i[2]) (instead of 3 it was written 2)
for i in villages_list:    
    if i[-2] == 1:
        set_rudimentary.add(i[3])
    elif i[-2] == 2:
        set_intermediate.add(i[3])
    elif i[-2] == 3:
        set_advance.add(i[3])
    else:
        continue

print(len(set_rudimentary))
print(len(set_advance))
print(len(set_intermediate))

from random import shuffle


for state_folder in os.listdir(images_path):

    print(state_folder)


    listfiles =  [images_name for images_name in os.listdir(os.path.join(images_path,state_folder))]
    shuffle(listfiles) #shuffling before diving into training and testing
    tottrain = (len(listfiles)*80) // 100
    trainfiles = listfiles[:tottrain]
    valfiles = listfiles[tottrain:]

    for images_name in trainfiles:

        image_code = images_name.split('@')[-1]
        image_code = image_code.split('.')[0]
        image_code = int(image_code)
        src = os.path.join(images_path,state_folder,images_name)

        if image_code in set_rudimentary:
            saveloc = os.path.join(save_path['train'][0])
            copy(src,saveloc)
        elif image_code in set_intermediate:
            saveloc = os.path.join(save_path['train'][1])
            copy(src,saveloc)
        elif image_code in set_advance:
            saveloc = os.path.join(save_path['train'][2])
            copy(src,saveloc)
        else:
            continue
          
          # Save in val folder
    for images_name in valfiles:
    
        image_code = images_name.split('@')[-1]
        image_code = image_code.split('.')[0]
        image_code = int(image_code)
        src = os.path.join(images_path,state_folder,images_name)

        if image_code in set_rudimentary:
            saveloc = os.path.join(save_path['val'][0])
            copy(src,saveloc)
        elif image_code in set_intermediate:
            saveloc = os.path.join(save_path['val'][1])
            copy(src,saveloc)
        elif image_code in set_advance:
            saveloc = os.path.join(save_path['val'][2])
            copy(src,saveloc)
        else:
            continue
          







  
