import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *

import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network.cuda()
# cudnn.benchmark = True

network.to(device)
if torch.cuda.is_available():
    cudnn.benchmark = True
    print("CUDA")
else:
    print("CPU")


epochs_saved = 0
checkpointFilename = 'network'
extension = '.pth'
CHECKPOINT = checkpointFilename+str(epochs_saved)+extension
RESULTS = "results/"

FOLDER = 'train'
# FOLDER = 'test'

# args.test = True
if not args.test:
    dataset = COCO("/scratch/lao/data/train/images/", "/scratch/lao/data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("/scratch/lao/data/train/images/", "/scratch/lao/data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    # dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    # dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    
    if os.path.exists(CHECKPOINT):
        network.load_state_dict(torch.load(CHECKPOINT,map_location=torch.device(device)))
        print("Loaded model to resume training")
    
    for epoch in range(num_epochs):
        #TRAIN
        network.train()
        
        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, shape = data
            # images = images_.cuda()
            # ann_box = ann_box_.cuda()
            # ann_confidence = ann_confidence_.cuda()
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            
            # nms_confidence, nms_box = non_maximum_suppression(pred_confidence[0].detach().cpu().numpy(), pred_box[0].detach().cpu().numpy(), boxs_default)
            
            #visualize
            # callVisualize(0,"train", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
            # callVisualize(1,"train", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
            
            # callVisualize(0,"train", nms_confidence, nms_box, ann_confidence_, ann_box_, images_, boxs_default)
            
            # for i in range(len(images_)):
            #     callVisualize(i,"train", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
            
            # pred_confidence[0].permute(2,0,1)
            # pred_box[0].permute(2,0,1)
            # images_[0].permute(1,2,0)
            
            # IMAGE_INDEX = 0
            # pred_confidence_ = pred_confidence[IMAGE_INDEX].detach().cpu().numpy()
            # pred_box_ = pred_box[IMAGE_INDEX].detach().cpu().numpy()
            # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[IMAGE_INDEX].numpy(), ann_box_[IMAGE_INDEX].numpy(), images_[IMAGE_INDEX].numpy(), boxs_default)
            
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
            
            # print('\rTraining: %d\t' % (i+1), end="")
            # print(avg_loss / avg_count, end="")
            
            # createTxt(True,i,pred_confidence, pred_box, shape, batch_size, boxs_default)
        
        print('\r[%d] time: %f \ttrain loss: %f\t\t\t' % (epoch+1, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        callVisualize(0,"train", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
        
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            # torch.save(network.state_dict(), CHECKPOINT)
            torch.save(network.state_dict(), checkpointFilename + str(epochs_saved+epoch+1) + extension)
            # for i in range(len(images_)):
            #     callVisualize(i,"train", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
    
    
    #TRAINING FINISHED
            
    #TEST
    network.eval()
    
    # TODO: split the dataset into 80% training and 20% testing
    # use the training set to train and the testing set to evaluate
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, shape = data
        # images = images_.cuda()
        # ann_box = ann_box_.cuda()
        # ann_confidence = ann_confidence_.cuda()
        images = images_.to(device)
        ann_box = ann_box_.to(device)
        ann_confidence = ann_confidence_.to(device)

        pred_confidence, pred_box = network(images)
        
        pred_confidence_ = pred_confidence.detach().cpu().numpy()
        pred_box_ = pred_box.detach().cpu().numpy()
        
        #optional: implement a function to accumulate precision and recall to compute mAP or F1.
        #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        print("\rTesting: %d\t\t\t" % (i+1), end='')
        
        #visualize
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        callVisualize(0,"test", pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default)
    
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)


else:
    #TEST
    
    test_batch_size = 1
    
    if FOLDER=='test':
        dataset_test = COCO("/scratch/lao/data/"+FOLDER+"/images/", None, class_num, boxs_default, train = False, image_size=320)
        # dataset_test = COCO("data/"+FOLDER+"/images/", None, class_num, boxs_default, train = False, image_size=320)
    else:
        dataset_test = COCO("/scratch/lao/data/"+FOLDER+"/images/", "data/"+FOLDER+"/annotations/", class_num, boxs_default, train = False, image_size=320)
        # dataset_test = COCO("data/"+FOLDER+"/images/", "data/"+FOLDER+"/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=0)
    
    if os.path.exists(CHECKPOINT):
        # network.load_state_dict(torch.load(CHECKPOINT))
        network.load_state_dict(torch.load(CHECKPOINT,map_location=torch.device(device)))
        print("Loaded model to resume training")
    else:
        print("No saved checkpoint for testing")
        sys.exit()
    
    network.eval()
    
    print("TESTING")
    
    RESULTS = RESULTS+FOLDER+'/'
    
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, shape = data
        # images = images_.cuda()
        # ann_box = ann_box_.cuda()
        # ann_confidence = ann_confidence_.cuda()
        images = images_.to(device)
        ann_box = ann_box_.to(device)
        ann_confidence = ann_confidence_.to(device)
        
        pred_confidence, pred_box = network(images)
        
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        
        # pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        nms_confidence, nms_box = non_maximum_suppression(pred_confidence[0].detach().cpu().numpy(), pred_box[0].detach().cpu().numpy(), boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        if FOLDER=='test':
            # createTxt(False,i,pred_confidence, pred_box, shape, test_batch_size, boxs_default)
            createTxt(False,i,nms_confidence, nms_box, shape, test_batch_size, boxs_default)
        else:
            # createTxt(True,i,pred_confidence, pred_box, shape, test_batch_size, boxs_default)
            createTxt(True,i,nms_confidence, nms_box, shape, test_batch_size, boxs_default)
        
        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        # cv2.waitKey(1000)
        
        callVisualize(0, FOLDER, pred_confidence, pred_box, ann_confidence_, ann_box_, images_, boxs_default, save=True,directory=RESULTS+str(i))
        callVisualize(0, FOLDER, nms_confidence, nms_box, ann_confidence_, ann_box_, images_, boxs_default, save=True,directory=RESULTS+str(i)+'nms')
        
        print('\rTesting: %d\t' % (i), end="")



