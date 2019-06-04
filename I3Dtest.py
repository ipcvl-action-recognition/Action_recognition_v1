import torch
import dataloader
import network
from torch.utils.data import DataLoader
from torch import optim
from dataloader import UCF_VideoDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2




def crop(self, buffer, clip_len, crop_size):
    time_index = np.random.randint(buffer.shape[1] - clip_len)
    height_index = np.random.randint(buffer.shape[2] - crop_size)
    width_index = np.random.randint(buffer.shape[3] - crop_size)
    # print(width_index, height_index)
    buffer = buffer[:, time_index:time_index + clip_len,
             height_index:height_index + crop_size,
             width_index:width_index + crop_size]
    return buffer


def normalize(self, buffer):
    buffer = (buffer - 128) / 128
    return buffer

if __name__=="__main__":
    path = 'D:/DATASET/UCF-101_recognition/'
    text = open(path+'classInd.txt', 'r')
    label = {}
    for line in text:
        line = line.strip()
        index, name = line.split(' ')
        label[int(index)-1] = name
    text.close()
    # print(label)
    # model = network.I3D(num_classes=101).cuda()
    resize_height = 224
    resize_width = 224

    fname = 'D:/DATASET/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        print("File is Not Open!!")
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # buffer = np.empty((frame_length, resize_height, resize_width, 3), np.dtype('float32'))
    buffer = []
    count = 0

    while count < frame_length and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imshow("frame", frame)
            cv2.waitKey(33)
            buffer.append(frame)
            if len(buffer) == 16:
                model = network.I3D(num_classes=101).cuda()
                input = np.array(buffer)
                input = np.transpose(input, (3, 0, 1, 2))
                input = torch.from_numpy(input)
                input = input.unsqueeze(0).float().cuda()
                output = model(input)[0]
                pred_index = torch.argmax(output)
                print(pred_index, label[pred_index.item()])
                # output
                buffer.pop(0)
            count += 1
        else:
            break

    cap.release()
    buffer = buffer.transpose((3, 0, 1, 2))  # c, l, w, h
    print("HI")