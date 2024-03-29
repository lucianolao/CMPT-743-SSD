import numpy as np
import cv2
from dataset import iou

import matplotlib.pyplot as plt
import math
import os
import torch


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image_ = (image_*255).astype(np.uint8)
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    shape = image.shape
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    layers = [10,5,3,1]
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                # image1, image2 = drawBox(ann_box, yind, xind, j, size, cellSize, shape, image1, image2)
                image1, image2 = drawBox(ann_box, i, j, shape, image1, image2, boxs_default)
    
    #pred
    # count = 0
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                # image3, image4 = drawBox(pred_box, yind, xind, j, size, cellSize, shape, image3, image4)
                # count = count+1
                image3, image4 = drawBox(pred_box, i, j, shape, image3, image4, boxs_default)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    return image


def non_maximum_suppression(confidence, box, boxs_default, overlap=0.2, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    N = len(boxs_default)
    class_num = confidence.shape[1]
    
    a_box = np.zeros([N,4], np.float32) #bounding boxes
    a_confidence = np.zeros([N,class_num], np.float32) #one-hot vectors
    a_confidence[:,-1] = 1 #the default class for all cells is set to "background"
    
    b_box = np.zeros([N,4], np.float32) #bounding boxes
    b_confidence = np.zeros([N,class_num], np.float32) #one-hot vectors
    b_confidence[:,-1] = 1 #the default class for all cells is set to "background"
    
    indices, categories = np.where(confidence[:,0:3] > threshold)
    
    a_confidence[indices] = confidence[indices]
    a_box[indices] = box[indices]
    
    # b_confidence = []
    # b_box = []
    
    # checkCategory = np.argmax(confidence[a], axis=1)
    
    # dx = a_box[:,0]
    # dy = a_box[:,1]
    # dw = a_box[:,2]
    # dh = a_box[:,3]
    
    
    dx = box[:,0]
    dy = box[:,1]
    dw = box[:,2]
    dh = box[:,3]
    
    px = boxs_default[:,0]
    py = boxs_default[:,1]
    pw = boxs_default[:,2]
    ph = boxs_default[:,3]
    
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    
    # centerX = gx * shape[1]
    # centerY = gy * shape[0]
    # width = gw * shape[1]
    # height = gh * shape[0]
    
    centerX = gx
    centerY = gy
    width = gw
    height = gh
    
    # x1 = centerX - (width/2)
    # y1 = centerY - (height/2)
    # x2 = centerX + (width/2)
    # y2 = centerY + (height/2)
    
    coords = np.zeros((len(a_box),8))
    
    # for i in range(len(a_box)):
    for i in range(len(boxs_default)):
        # coords[i,0] = max(centerX[i] - (width[i]/2), 0) # x1
        # coords[i,1] = max(centerY[i] - (height[i]/2), 0) # y1
        # coords[i,2] = min(centerX[i] + (width[i]/2), 1) # x2
        # coords[i,3] = min(centerY[i] + (height[i]/2), 1) # y2
        
        coords[i,0] = centerX[i]
        coords[i,1] = centerY[i]
        coords[i,2] = width[i]
        coords[i,3] = height[i]
        coords[i,4] = centerX[i] - (width[i]/2) # x1
        coords[i,5] = centerY[i] - (height[i]/2) # y1
        coords[i,6] = centerX[i] + (width[i]/2) # x2
        coords[i,7] = centerY[i] + (height[i]/2) # y2
    
    
    # coords[:,0] = max(centerX[:] - (width[:]/2), 0) # x1
    # coords[:,1] = max(centerY - (height/2), 0) # y1
    # coords[:,2] = min(centerX + (width/2), 1) # x2
    # coords[:,3] = min(centerY + (height/2), 1) # y2
    
 
    
    # indices_carrying, values_carrying = np.where(pred_confidence[i,:,0:3] > 0.5)
    
    # obj_detected = torch.argmax(pred_confidence[i],1)
    # indices_filled = (obj_detected != 3).nonzero()
    # indices_filled = indices_filled.reshape(len(indices_filled))
    
    # indices_carrying, values_carrying = np.where(confidence[a,0:3] > 0.5)
    
    
    # while (confidence[row,col] > threshold):
    # while (a_confidence != None):
    while (np.max(a_confidence[:,0:3]) > threshold):
        position = np.argmax(a_confidence[:,0:3])
        row, col = np.unravel_index(position, [N, 3])
        # col = position % 3
        # row = math.floor(position/3)
        
        category = col
        x = row
        
        # Copy A to B
        # b_confidence.append(a_confidence[x])
        # b_box.append(a_box[x])
        b_confidence[x,:] = a_confidence[x,:]
        b_box[x,:] = a_box[x,:]
        
        # Remove from A
        # a_box.remove(x)
        # a_confidence = np.delete(a_confidence, x, axis=0)
        # a_box = np.delete(a_box, x, axis=0)
        a_confidence[x,:] = [0,0,0,1]
        a_box[x,:] = [0,0,0,0]
        
        
        x_min = coords[x,4]
        y_min = coords[x,5]
        x_max = coords[x,6]
        y_max = coords[x,7]
        
        # indices_carrying, values_carrying = np.where(confidence[a,category] > 0.5)
        
        # indices_to_check = np.where(a_confidence[:,category] > threshold)[0]
        indices_to_check = np.where(a_confidence[:,0:3] > threshold)[0]
        
        # if len(indices_to_check) > 0:
        #     # iou_results = iou(boxs_default[indices_to_check], x_min,y_min,x_max,y_max)
        #     iou_results = iou(coords[indices_to_check], x_min,y_min,x_max,y_max)
        #     if len(iou_results > 0):
        #         iou_indices = np.where(iou_results > overlap)[0]
        #         indices_to_delete = indices_to_check[iou_indices]
        #         a_confidence[indices_to_delete,:] = [0,0,0,1]
        #         a_box[indices_to_delete,:] = [0,0,0,0]
        
        iou_results = iou(coords[indices_to_check], x_min,y_min,x_max,y_max)
        iou_indices = np.where(iou_results > overlap)[0]
        
        indices_to_delete = indices_to_check[iou_indices]
        a_confidence[indices_to_delete,:] = [0,0,0,1]
        a_box[indices_to_delete,:] = [0,0,0,0]
        
        # for element in a:
        #     if confidence[element,category] > threshold:
        #         # x_min, y_min, x_max, y_max, _, _ = boxToImage(box[x], [1,1], boxs_default[x])
        #         x_min = coords[x,0]
        #         y_min = coords[x,1]
        #         x_max = coords[x,2]
        #         y_max = coords[x,3]
        #         iou_result = iou(boxs_default[x], x_min,y_min,x_max,y_max)
        #         if iou_result > overlap:
        #             # Remove x from from a
        #             a.remove(x)
        
        

    
    #TODO: non maximum suppression
    return b_confidence, b_box


def boxToImage(box_row, shape, boxs_default_row):
    dx = box_row[0]
    dy = box_row[1]
    dw = box_row[2]
    dh = box_row[3]
    
    px = boxs_default_row[0]
    py = boxs_default_row[1]
    pw = boxs_default_row[2]
    ph = boxs_default_row[3]
    
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * math.exp(dw)
    gh = ph * math.exp(dh)
    
    centerX = gx * shape[1]
    centerY = gy * shape[0]
    width = gw * shape[1]
    height = gh * shape[0]
    
    x1 = int(centerX - (width/2))
    y1 = int(centerY - (height/2))
    x2 = int(centerX + (width/2))
    y2 = int(centerY + (height/2))
    
    return x1,y1,x2,y2,width,height


def drawBox(box, i, j, shape, imageL, imageR, boxs_default):
    
    color = colors[j]
    thickness = 2
    
    x1,y1,x2,y2,_,_ = boxToImage(box[i], shape, boxs_default[i])
    
    start_point = (x1, y1)
    end_point = (x2, y2)
    
    cv2.rectangle(imageL, start_point, end_point, color, thickness)
    
    
    x1 = int(boxs_default[i,4] * shape[1])
    y1 = int(boxs_default[i,5] * shape[0])
    x2 = int(boxs_default[i,6] * shape[1])
    y2 = int(boxs_default[i,7] * shape[0])
    
    start_point = (x1, y1)
    end_point = (x2, y2)
    
    cv2.rectangle(imageR, start_point, end_point, color, thickness)
    
    return imageL, imageR
    
    
def callVisualize(index,nameWindow, pred_confidence, pred_box, ann_confidence, ann_box, images, boxs_default, save=False,directory=""):
    i = index
    if torch.is_tensor(pred_confidence):
        pred_confidence = pred_confidence[i].detach().cpu().numpy()
    if torch.is_tensor(pred_box):
        pred_box = pred_box[i].detach().cpu().numpy()
    if torch.is_tensor(ann_confidence):
        ann_confidence = ann_confidence[i].detach().cpu().numpy()
    if torch.is_tensor(ann_box):
        ann_box = ann_box[i].detach().cpu().numpy()
    if torch.is_tensor(images):
        images = images[i].detach().cpu().numpy()
    img_BGR = visualize_pred(nameWindow, pred_confidence, pred_box, ann_confidence, ann_box, images, boxs_default)
    if save:
        # img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory + ".jpeg", img_BGR)
    else:
        # cv2.imshow(nameWindow+" [[gt_box,gt_dft],[pd_box,pd_dft]]",img_BGR)
        # cv2.waitKey(1000)
        plt.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB))
        plt.title(directory[8:])
        # plt.axis('off')
        # plt.savefig(directory + '.jpeg')
        plt.show()


def createTxt(isTraining, iteration, pred_confidence, pred_box, shape, batch_size, boxs_default):
    if isTraining:
        directory = "predicted_boxes/train/"
    else:
        directory = "predicted_boxes/test/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # filename = os.path.join(directory, "%05d"%(index) + '.txt')
    
    
    
    if torch.is_tensor(pred_box):
        pred_box = pred_box.detach().cpu().numpy()
    if torch.is_tensor(pred_confidence):
        pred_confidence = pred_confidence.detach().cpu().numpy()
        
    if pred_box.ndim == 2:
        pred_box = np.reshape(pred_box, (batch_size, pred_box.shape[0], pred_box.shape[1]))
    
    if pred_confidence.ndim == 2:
        pred_confidence = np.reshape(pred_confidence, (batch_size, pred_confidence.shape[0], pred_confidence.shape[1]))
    
    current_batch_size = len(pred_box)
    
    for i in range(current_batch_size):
        imageID = iteration*batch_size + i
        filename = os.path.join(directory, "%05d"%(imageID) + '.txt')
        with open(filename,"w") as f:
            # obj_detected = torch.argmax(pred_confidence[i],1)
            # indices_filled = (obj_detected != 3).nonzero()
            # indices_filled = indices_filled.reshape(len(indices_filled))
            
            # obj_detected = np.argmax(pred_confidence[i],1)
            indices_carrying, values_carrying = np.where(pred_confidence[i,:,0:3] > 0.5)
            # indices_filled = (values != 3).nonzero()
            
            number_objects = len(indices_carrying)
            for j in range(number_objects):
                # SORT HERE IF NEEDED
                index = indices_carrying[j]
                cat_id = values_carrying[j]
                x1,y1,_,_,width,height = boxToImage(pred_box[i,index], shape[i].numpy(), boxs_default[index])
                content = str(cat_id) +' '+ str('%.1f'%float(x1)) +' '+ str('%.1f'%float(y1)) +' '+ str('%.2f'%float(width)) +' '+ str('%.2f'%float(height)) + '\n'
                f.write(content)
    # print("what")
    # print("ever")


def plot(image):
    plt.imshow(image)
    plt.show()