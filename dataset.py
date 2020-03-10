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
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
import math

#generate default bounding boxes
# boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    
    
    squared_layers = np.square(layers) # [100,25,9,1]
    n_cells = sum(squared_layers) # 135
    n_boxes_per_cell = len(layers) # 4
    box_num = n_cells*n_boxes_per_cell # 540
    boxes3D = np.zeros((n_cells,n_boxes_per_cell,8)) # (135,4,8)
    
    repeated = [] # [100,100,100,100...25,25,25...9,9...1]
    accumulation = [0]*4 # [0, 100, 125, 134]
    summ=0
    for i in range(len(layers)):
        repeated = np.concatenate((repeated,np.repeat(layers[i],squared_layers[i],axis=0)))
        accumulation[i] = summ
        summ = summ + squared_layers[i]
    
    for i in range(n_cells):
        size = repeated[i] # 10 || 5 || 3 || 1
        cellSize = 1/size
        layer_index = layers.index(size)
        relative_i = i - accumulation[layer_index]
        cellX = relative_i % size
        cellY = math.floor(relative_i / size)
        for j in range(n_boxes_per_cell):
            center_x = (cellX * cellSize) + cellSize/2
            center_y = (cellY * cellSize) + cellSize/2
            boxes3D[i][j][0] = center_x
            boxes3D[i][j][1] = center_y
            if j==0:
                width = small_scale[layer_index]
                height = width
            elif j==1:
                width = large_scale[layer_index]
                height = width
            elif j==2:
                width = large_scale[layer_index] * math.sqrt(2)
                height = large_scale[layer_index] / math.sqrt(2)
            elif j==3:
                width = large_scale[layer_index] / math.sqrt(2)
                height = large_scale[layer_index] * math.sqrt(2)
            boxes3D[i][j][2] = width
            boxes3D[i][j][3] = height
            boxes3D[i][j][4] = max(center_x - width/2, 0)    # x_min
            boxes3D[i][j][5] = max(center_y - height/2, 0)    # y_min
            boxes3D[i][j][6] = min(center_x + width/2, 1)    # x_max
            boxes3D[i][j][7] = min(center_y + height/2, 1)    # y_max
    
    boxes = boxes3D.reshape((box_num,8)) # 540,8
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    gw = x_max - x_min
    gh = y_max - y_min
    gx = x_min + gw/2
    gy = y_min + gh/2
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    # indices = (ious_true == True).nonzero()
    indices = np.where(ious_true)[0]
    # indices = indices[0]
    if len(indices) == 0:
        indices = [np.argmax(ious)]
    for i in range(len(indices)):
        px = boxs_default[indices[i]][0]
        py = boxs_default[indices[i]][1]
        pw = boxs_default[indices[i]][2]
        ph = boxs_default[indices[i]][3]
        
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = math.log(gw/pw, 10)
        th = math.log(gh/ph, 10)
        
        ann_box[indices[i]][0] = tx
        ann_box[indices[i]][1] = ty
        ann_box[indices[i]][2] = tw
        ann_box[indices[i]][3] = th
        
        ann_confidence[indices[i]][cat_id] = 1
        ann_confidence[indices[i]][-1] = 0
        
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    # ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    # a=1


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.img_names.sort()
        self.image_size = image_size
        
        PERCENT_FOR_TRAINING = 0.8
        
        total = len(self.img_names)
        
        partition = round(total * PERCENT_FOR_TRAINING)
        
        if self.train:
            self.img_names = self.img_names[0:partition]
            print("DATASET: split training")
        else:
            self.img_names = self.img_names[partition:total]
            print("DATASET: split testing")
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        if self.anndir != None:
            ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        # original = cv2.imread("data/train/images/00070.jpg")
        original = cv2.imread(img_name)
        # originalRGB = plt.imread(img_name)
        shape = original.shape
        # ratioHeight = self.image_size / shape[0]
        # ratioWidth = self.image_size / shape[1]
        # dim = (width, height)
        dim = (self.image_size, self.image_size)
        image = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
        image = np.swapaxes(image,2,1)
        image = np.swapaxes(image,1,0)
        
        
        if self.anndir == None:
            image = image/255.0
            image = image.astype('float32')
            return image, ann_box, ann_confidence, torch.Tensor(shape)
        
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # with open("data/train/annotations/00070.txt") as f:
        with open(ann_name) as f:
            content = f.read().splitlines()
        
        
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        for i in range(0,len(content)):
            obj = content[i].split(" ")
            cat_id = int(obj[0])
            # centerX = float(obj[1]) / shape[1]
            # centerY = float(obj[2]) / shape[0]
            # width = float(obj[3]) / shape[1]
            # height = float(obj[4]) / shape[0]
            
            # x_min = centerX - width/2
            # x_max = centerX + width/2
            # y_min = centerY - height/2
            # y_max = centerY + height/2
            
            
            x_min = float(obj[1]) / shape[1]
            y_min = float(obj[2]) / shape[0]
            width = float(obj[3]) / shape[1]
            height = float(obj[4]) / shape[0]
            
            x_max = x_min + width
            y_max = y_min + height
            

            # match(ann_box,ann_confidence,cat_id,x_min,y_min,x_max,y_max)
            match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,x_min,y_min,x_max,y_max)
        
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        image = image/255.0
        image = image.astype('float32')
        
        return image, ann_box, ann_confidence, torch.Tensor(shape)
    
    def plot(image):
        plt.imshow(image)
        plt.show()
