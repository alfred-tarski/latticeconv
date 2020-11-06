import os
from convert_point_cloud import off_to_pointcloud
from features import density_filtration
from rivet import write_rivet_input,rivet_find_invariants,parse_rivet_output
import numpy as np
import numpy.random as rd
import torch
import multiprocessing as mp

main_path = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"

def proccess_file(filename,path_train,index,max_size=1000,k=100,max_dist = 10e6,degs=1,x_bins =40,y_bins=40):
    if filename == ".DS_Store":
        continue
    path = path_train + filename
    print(filename)
    point_cloud = off_to_pointcloud(path)
    if point_cloud.shape[0]>max_size:
        indices = rd.randint(0,point_cloud.shape[0],max_size)
        point_cloud = point_cloud[indices,:]
    name = "/home/ubuntu/code/rivet-input/training_" + str(category)+"_" + str(index)+".txt"
    write_rivet_input(point_cloud,density_filtration,k,max_dist,name)
    for deg in range(degs):
        output_name = '/home/ubuntu/code/invariants/training_'+str(category)+ "_h"+ str(deg)+"_"+str(index)+".txt"
        rivet_find_invariants(name,output_name,deg,x_bins,y_bins)

categories = rd.shuffle(os.listdir(main_path))
for category in categories:
    path_train = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"+category+'/train/'
    [pool.apply(proccess_file,args=(filename,path_train,index)) for index,filename in enumerate(os.listdir(path_train))]