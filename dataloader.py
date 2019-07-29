import numpy as np
import cv2
import torch
import xml.etree.ElementTree
import os
from torch.utils.data import DataLoader
from ImageArg import ImageArg
import time

class UCF101_Dataset:
    def __init__(self, folder_path, readfile, is_train=True):
        self.folder_path = folder_path
        folder_list = os.listdir(folder_path)
        self.folder_list = folder_list

        data_list = []
        file = open(readfile, 'r')

        for line in file:
            line = line.split(' ')[0].strip()
            line = folder_path.strip() + '/' + line
            data_list.append(line)

        self.video_list = data_list
        self.clip_len = 16
        self.crop_size = 112
        self.resize_width = 171
        self.resize_height = 128
        self.data_length = len(self.video_list)

        self.imgArg = ImageArg()
        self.allImage_buffer_list = []
        self.is_train = is_train
        # self.loadAllVideo()

    def loadAllVideo(self):
        for idx in range(len(self.video_list)):
            self.allImage_buffer_list.append([self.loadvideo(self.video_list[idx])])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        buffer = self.loadvideo(self.video_list[idx])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        label = self.video_list[idx].split('/')[-2]
        label = self.folder_list.index(label)
        return torch.from_numpy(buffer), label

    def loadvideo(self, fname):
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            print("File is Not Open!!")
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('int8'))

        count = 0
        # print(frame_length)
        while count < frame_length and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer[count] = frame
                count += 1

            else:
                break

        cap.release()
        buffer = buffer.transpose((3, 0, 1, 2))  # c, l, h, w
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
        # buffer = buffer.transpose((1, 2, 3, 0))  # f, h, w, c
        buffer = buffer.astype(np.float32)
        # for i, frame in enumerate(buffer):
        #    frame -= np.array([[[90.0, 98.0, 102.0]]])
        #    buffer[i] = frame
        buffer = buffer / 255.0  # (buffer-128)/128
        # buffer = buffer.transpose((3, 0, 1, 2))
        return buffer

    def label_extraction(self, fname, startFrame):
        label_list = self.framefile_path + "/" + fname
        readlabel = open(label_list, 'r')
        label_list = []
        for i, line in enumerate(readlabel):
            line = line.strip()
            line = line.split(' ')
            label_list.append(line[1])

        if self.is_train:
            label_list = label_list[startFrame:startFrame + self.clip_len]
            label = int(label_list[-1])
            return label
        else:
            return [''.join(label_list)]


class Le2i_VideoDataset:
    def __init__(self, folder_path, readfile, framefile, is_train=True):
        self.folder_path = folder_path
        folder_list = os.listdir(folder_path)
        self.folder_list = folder_list

        data_list = []
        file_list = open(readfile, 'r')
        for line in file_list:
            line = line.strip()
            data_list.append(folder_path + "/" + line)

        self.framefile_path = framefile
        frame_folder_list = os.listdir(framefile)
        self.frame_label_list = frame_folder_list

        self.video_list = data_list
        self.clip_len = 16
        # self.crop_size = 224
        self.resize_width = 112
        self.resize_height = 112
        self.data_length = len(self.video_list)

        self.imgArg = ImageArg()
        self.allImage_buffer_list = []
        self.is_train = is_train
        # self.loadAllVideo()

    def loadAllVideo(self):
        for idx in range(len(self.video_list)):
            self.allImage_buffer_list.append([self.loadvideo(self.video_list[idx])])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if self.is_train:
            buffer, frame_length = self.loadvideo(self.video_list[idx])
            # buffer = np.squeeze(np.array(self.allImage_buffer_list[idx]))
            # buffer, startFrame = self.crop(buffer, self.clip_len, self.crop_size)
            startFrame=None
            if self.is_train:
                buffer, startFrame = self.crop(buffer, self.clip_len, frame_length)

            # self.imgArg.ImageArgument(buffer)
            if self.is_train:
                buffer = self.imgArg.ImageArgument(buffer)
            buffer = self.normalize(buffer)
            label = self.label_extraction(self.frame_label_list[idx], startFrame)
            return torch.from_numpy(buffer), label, frame_length
        else:
            return self.video_list[idx]

    def loadvideo(self, fname):
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            print("File is Not Open!!")
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.is_train :
            frame_length = 159
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer = np.empty((frame_length, self.resize_height, self.resize_width, 3), np.dtype('int8'))

        count = 0
        while count < frame_length and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if count == 1500:
                    count = 0
                buffer[count] = frame
                count += 1

            else:
                break
        count=0
        cap.release()
        buffer = buffer.transpose((3, 0, 1, 2))  # c, l, h, w
        return buffer, frame_length

    # def crop(self, buffer, clip_len, crop_size):
    def crop(self, buffer, clip_len, frame_length):
        time_index = np.random.randint(frame_length - clip_len)
        # height_index = np.random.randint(buffer.shape[2] - crop_size)
        # width_index = np.random.randint(buffer.shape[3] - crop_size)

        # buffer = buffer[:, time_index:time_index + clip_len,
        #         height_index:height_index + crop_size,
        #         width_index:width_index + crop_size]
        buffer = buffer[:, time_index:time_index + clip_len]

        return buffer, time_index

    def normalize(self, buffer):
        # buffer = buffer.transpose((1, 2, 3, 0))  # f, h, w, c
        buffer = buffer.astype(np.float32)
        # for i, frame in enumerate(buffer):
        #    frame -= np.array([[[90.0, 98.0, 102.0]]])
        #    buffer[i] = frame
        buffer = buffer / 255.0  # (buffer-128)/128
        # buffer = buffer.transpose((3, 0, 1, 2))
        return buffer

    def label_extraction(self, fname, startFrame):
        label_list = self.framefile_path + "/" + fname
        readlabel = open(label_list, 'r')
        label_list = []
        for i, line in enumerate(readlabel):
            line = line.strip()
            line = line.split(' ')
            label_list.append(line[1])

        if self.is_train:
            label_list = label_list[startFrame:startFrame + self.clip_len]
            label = int(label_list[-1])
            return label
        else:
            return [''.join(label_list)]
