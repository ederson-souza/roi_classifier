# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import pafy


from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model
from bounding_box import bounding_box as bb


if __name__ == '__main__':

  # Current Directory
  CUR_DIR = os.getcwd()

  # load video from YouTube
  url = 'https://www.youtube.com/watch?v=PJ5xXXcfuTc&ab_channel=Supercircuits'
  vPafy = pafy.new(url)
  play = vPafy.getbest() # get best resolution

  cap = cv2.VideoCapture(play.url) # video capture

  # Get video resolution
  width = cap.get(3)
  height = cap.get(4)

  # Get frames' total
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

  # Set starting frame
  num_frame = 360

  # Model Configuration
  transform_fn = transforms.Compose([
      transforms.Resize(32),
      transforms.CenterCrop(32),
      transforms.ToTensor(),
      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
  ])

  # Setting a ResNet as CNN
  net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)

  # Naming the classes
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


  # Output Configuration
  try:
    os.mkdir(os.path.join(CUR_DIR, 'output'))
  except:
    pass

    # Path to the outup file
  OUTPUT_PATH = os.path.join(CUR_DIR, 'output/result.mp4')
  
    # Video settings
  video = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(width),int(height)),True)
  
  num_start_frame = num_frame
  
  # Start Classification 
  first = True

  while(cap.isOpened()):
    
    if num_frame > total_frames:
      break
    
    try:
      cap.set(1, num_frame)  
      _ , frame = cap.read() #read frame

      if first: # if first frame, open Roi Selector
        r = cv2.selectROI(frame)
        
        # get coordinates
        roi_x1 = int(r[0])
        roi_x2 = int(r[0] + r[2])
        roi_y1 = int(r[1])
        roi_y2 = int(r[1]+r[3])
        cv2.destroyAllWindows()
        first = False
      
      # Print progress 
      print(f'{(num_frame-num_start_frame)}/{int(total_frames-num_start_frame)} - {((num_frame-num_start_frame)/(total_frames-num_start_frame)*100):.2f}%', end='\r')
      
      # Crop Roi
      crop_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
      
      # Save a temporary file
      cv2.imwrite('crop.jpg', crop_frame)
      
      # Read and transform the image
      crop_frame = image.imread('crop.jpg')
      crop_frame = transform_fn(crop_frame)

      # Make Prediction
      output = net(crop_frame.expand_dims(axis=0))
      
      # Get the most probable class index
      index = nd.argmax(output, axis=1).astype('int')
      
      # Get the class's probability and convert to a formatted string
      prob = f'{nd.softmax(output)[0][index].asscalar()*100:.2f}%'

      # Check whether the most probable class is automobile
      if class_names[index.asscalar()] == 'automobile':
        class_ = 'car'
      else:
        class_ = 'not a car'

    except:
      break
    
  # Draw ROI and prediction label
    bb.add(frame, roi_x1, roi_y1, roi_x2, roi_y2, class_ + '-' + prob, 'yellow')

  # Record the output frame
    video.write(frame)

  # Go to the next frame
    num_frame += 1  
   

  cap.release()
  video.release()
  os.remove("crop.jpg")
