import os
import pickle
year = "2018"
path = "/home/cse/mtech/mcs202448/scratch/Split_Data/"+year+"_Split"
# states = ['PB','CG', 'UP', 'MH', 'AP', 'BR', 'JH', 'GJ', 'TN', 'KA', 'RJ', 'HR', 'MP', 'OR']
states = ['PB']
l = []
count = 0
for state in states:
    for vill_name in os.listdir(path+"/"+state+"_150x/"):
        l.append(state+"_150x/"+vill_name)
        count += 1
        if (count > 200):
            break
print("Number of Villages = ", count)

with open('Village_Lists/List_Villages_Punjab'+year+'.pkl', 'wb') as f:
    pickle.dump(l,f)