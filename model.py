import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    batch_size = ann_confidence.shape[0]
    num_of_boxes = ann_confidence.shape[1]
    num_of_classes = ann_confidence.shape[2]
    
    # Reshaping
    pred_confidence = pred_confidence.reshape(batch_size*num_of_boxes, num_of_classes)
    pred_box = pred_box.reshape(batch_size*num_of_boxes, 4)
    ann_confidence = ann_confidence.reshape(batch_size*num_of_boxes, num_of_classes)
    ann_box = ann_box.reshape(batch_size*num_of_boxes, 4)
    
    # Getting indices of boxes with objects
    _, obj_detected = torch.max(ann_confidence, 1)
    indices_filled = (obj_detected != 3).nonzero()
    indices_empty = (obj_detected == 3).nonzero()
    
    # Removing second dimension of size 1
    indices_filled = indices_filled.reshape(len(indices_filled))
    indices_empty = indices_empty.reshape(len(indices_empty))
    
    # Boxes and Confidence carrying objects
    pred_confidence_carrying = pred_confidence[indices_filled]
    pred_box_carrying = pred_box[indices_filled]
    ann_confidence_carrying = ann_confidence[indices_filled]
    ann_box_carrying = ann_box[indices_filled]
    
    # Confidence without objects
    pred_confidence_empty = pred_confidence[indices_empty]
    ann_confidence_empty = ann_confidence[indices_empty]
    
    loss_cls = F.binary_cross_entropy(pred_confidence_carrying, ann_confidence_carrying)
    loss_cls = loss_cls + 3*F.binary_cross_entropy(pred_confidence_empty, ann_confidence_empty)
    loss_box = F.smooth_l1_loss(pred_box_carrying, ann_box_carrying)
    loss = loss_cls + loss_box
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    return loss


def conv_bat_re(cin, cout, ksize, s, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=ksize, stride=s, padding=p),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True)
    )


def permute(x):
    x = x.permute(0,2,1)
    return x


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.layer = nn.ModuleList()
        
        self.layer.append(conv_bat_re(3, 64, 3, 2))
        
        self.layer.append(conv_bat_re(64, 64, 3, 1))
        self.layer.append(conv_bat_re(64, 64, 3, 1))
        self.layer.append(conv_bat_re(64, 128, 3, 2))
        
        self.layer.append(conv_bat_re(128, 128, 3, 1))
        self.layer.append(conv_bat_re(128, 128, 3, 1))
        self.layer.append(conv_bat_re(128, 256, 3, 2))
        
        self.layer.append(conv_bat_re(256, 256, 3, 1))
        self.layer.append(conv_bat_re(256, 256, 3, 1))
        self.layer.append(conv_bat_re(256, 512, 3, 2))
        
        self.layer.append(conv_bat_re(512, 512, 3, 1))
        self.layer.append(conv_bat_re(512, 512, 3, 1))
        self.layer.append(conv_bat_re(512, 256, 3, 2))
        
        # 13 layers above
        
        self.layer.append(conv_bat_re(256, 256, 1, 1, 0))
        self.layer.append(conv_bat_re(256, 256, 3, 2))
        
        # Different than YOLO from here
        self.layer.append(conv_bat_re(256, 256, 1, 1, 0))
        self.layer.append(conv_bat_re(256, 256, 3, 1, 0))
        
        self.layer.append(conv_bat_re(256, 256, 1, 1, 0))
        self.layer.append(conv_bat_re(256, 256, 3, 1, 0))
        
        self.red4 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        self.blue4 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        
        self.red1 = lastConvs(100)
        self.blue1 = lastConvs(100)
        
        self.red2 = lastConvs(25)
        self.blue2 = lastConvs(25)
        
        self.red3 = lastConvs(9)
        self.blue3 = lastConvs(9)
        
        # self.box = boxPath()
        # self.conf = confidencePath(class_num)
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        
        for i in range(13): # 0-12
            x = self.layer[i](x)
        
        red1 = self.red1(x)
        blue1 = self.blue1(x)
        
        x = self.layer[13](x)
        x = self.layer[14](x)
        
        red2 = self.red2(x)
        blue2 = self.blue2(x)
        
        x = self.layer[15](x)
        x = self.layer[16](x)
        
        red3 = self.red3(x)
        blue3 = self.blue3(x)
        
        x = self.layer[17](x)
        x = self.layer[18](x)
        
        red4 = self.red4(x)
        red4 = red4.reshape(len(x),16,1)
        blue4 = self.blue4(x)
        blue4 = blue4.reshape(len(x),16,1)
        
        red = torch.cat([red1,red2,red3,red4],dim=2)
        blue = torch.cat([blue1,blue2,blue3,blue4], dim=2)
        
        red = permute(red)
        blue = permute(blue)
        
        red = red.reshape(len(x), 540, 4)
        blue = blue.reshape(len(x), 540, self.class_num)
        
        bboxes = red
        
        blue = torch.softmax(blue,dim=2)
        confidence = blue
        
        return confidence,bboxes

class lastConvs(nn.Module):
    def __init__(self, newShape):
        super(lastConvs, self).__init__()
        self.conv = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.newShape = newShape
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(len(x),16,self.newShape)
        return x


# class confidencePath(nn.Module):
#     def __init__(self, class_num):
#         super(confidencePath, self).__init__()
#         self.conv1 = nn.Conv2d(256, class_num, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         # x = F.softmax(x,dim=1)
#         x = torch.softmax(x,dim=1)
#         x = permute(x)
#         return x


# class boxPath(nn.Module):
#     def __init__(self):
#         super(boxPath, self).__init__()
#         self.conv1 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         # x = F.relu(x)
#         x = permute(x)
#         return x


# class conv_bat_re(nn.Module):
#     # def __init__(self, inC, outC):
#     def __init__(self, cin, cout, ksize, s, p=1):
#         super(conv_bat_re, self).__init__()
#         self.conv1 = nn.Conv2d(cin, cout, kernel_size=ksize, stride=s, padding=p)
#         self.batch = nn.BatchNorm2d(cout)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch(x)
#         x = self.relu(x)
#         # x = F.relu(x)
#         return x






