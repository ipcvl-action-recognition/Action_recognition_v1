from networks import *
import models
import cv2
import torch
import numpy as np


video_file = 'D:/fire1.mp4'
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
print("a")
