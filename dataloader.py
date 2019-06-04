import numpy as np
import cv2
import torch
import xml.etree.ElementTree
import os
class UCF_VideoDataset:
    def __init__(self, folder_path, readfile):
        self.folder_path = folder_path
        folder_list = os.listdir(folder_path)
        self.folder_list = folder_list
        data_list = []
        file_list = open(readfile, 'r')
        for line in file_list:
            line = line.strip()
            line = line.split(' ')
            data_list.append(folder_path+"/"+line[0])
        file_list.close()
        one_hot_vector = np.eye(len(self.folder_list))
        self.video_list = data_list
        self.clip_len = 16
        self.crop_size = 112
        self.resize_width = 171
        self.resize_height = 128

    def __getitem__(self, idx):
        # idx = 0
        buffer = self.loadvideo(self.video_list[idx])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        # Label
        label = self.video_list[idx].split('/')[3]
        label = self.folder_list.index(label)

        buffer = torch.from_numpy(buffer)
        label = np.asarray(label)
        label = torch.from_numpy(label)
        return buffer, label

    def loadvideo(self, fname):
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            print("File is Not Open!!")
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0

        while count < frame_length and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                # cv2.imshow("frame", frame)
                # cv2.waitKey(33)
                buffer[count] = frame
                count += 1
            else:
                break

        cap.release()
        buffer = buffer.transpose((3, 0, 1, 2))  # c, l, w, h
        return buffer

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
        buffer = (buffer-128)/128
        return buffer

    def __len__(self):
        return len(self.video_list)

if __name__=='__main__':
    folder_path = 'D:/DATASET/UCF-101'
    readfile = 'D:/DATASET/UCF-101_recognition'
    UCF_VideoDataset(folder_path=folder_path, readfile=readfile+'/trainlist01.txt')