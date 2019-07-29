import torch
import numpy as np
from network import C3DNet
import cv2
from torch import nn

torch.backends.cudnn.benchmark = True
import time


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def frame_labeling(filename):
    file = open(filename, 'r')
    label_list = []
    for i, line in enumerate(file):
        line = line.strip()
        line = line.split(' ')
        label_list.append(line[1])
    return label_list
def test_accuracy(is_validation=False, video=None, folder_path=None):
    model = C3DNet().cuda()
    criterion = torch.nn.BCELoss().cuda()

    if not is_validation:
        checkpoint = torch.load('Weights/Falldown_27_0.51_0.32_85.96.pt', map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    model.eval()

    video_loss = []
    # for video in video_list:
    cap = cv2.VideoCapture(video)
    retaining = True
    clip = []
    video = video.replace("Video", "frame_label")
    filename = video.replace("mp4", "txt")
    # label 부분은 함수로 뺴서 따로 처리해서 만들면 될듯
    label = frame_labeling(filename)
    frame_number = 0
    correct = 0
    val_epoch_losses = []
    total_len = 0
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp = cv2.resize(frame, (112, 112))
        tmp = tmp.astype(np.float32)
        tmp = tmp / 255.0
        clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)  # input_shape =  (16, 112, 112, 3)
            inputs = np.expand_dims(inputs, axis=0)  # input_shape =  (1, 16, 112, 112, 3)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))  # input_shape =  (1, 3, 16, 112, 112)
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()
            with torch.no_grad():
                outputs = model.forward(inputs)

            if not is_validation:
                outputs = np.squeeze(torch.sigmoid(outputs).cpu().numpy())
                output_frame = cv2.resize(frame, dsize=(640, 480))
                if (outputs > 0.5):
                    prob = outputs
                    cv2.putText(output_frame, "FallDown(prob: %.4f)" % prob, (20, 40),
                                cv2.FONT_HERSHEY_COMPLEX, 1.2,
                                (0, 0, 255), 2)
                else:
                    prob = 1.0 - outputs
                    cv2.putText(output_frame, "None(prob: %.4f)" % prob, (20, 40),
                                cv2.FONT_HERSHEY_COMPLEX, 1.2,
                                (255, 0, 0), 2)


                cv2.imshow('result', output_frame)
                cv2.waitKey(1)
            else:
                y_pred = torch.sigmoid(outputs)
                y_frame = torch.from_numpy(np.array([label[frame_number + 15]]).astype(int)).cuda()
                # print(y_frame)
                loss = criterion(y_pred.squeeze(0), y_frame.float()).cuda()
                y_pred_index = torch.round(y_pred).int()
                y_pred_index = torch.transpose(y_pred_index, 0, 1)
                y_frame = y_frame.int()
                correct += (y_pred_index == y_frame).sum().item()
                val_epoch_losses.append(loss.item())

            clip.pop(0)
            frame_number+=1
            total_len += 1
    # video_loss.append(np.mean(val_epoch_losses))
    accuracy = correct / total_len * 100
    if is_validation:
        return np.mean(val_epoch_losses), accuracy
def main():
    # init model
    model = C3DNet().cuda()

    checkpoint = torch.load('Weights/Falldown_27_0.51_0.32_85.96.pt', map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint)

    # model.to(device)
    model.eval()

    # read video
    # video = '/cam/cam2.avi'
    # video = './Test_data/Video/Train/Coffee_room_01 (02).mp4'
    video = './Test_data/Video/Val/Coffee_room_01 (10).mp4'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []

    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        tmp_ = cv2.resize(frame, (224, 224))
        # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        tmp = tmp_.astype(np.float32)
        tmp = tmp / 255.0

        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)  # input_shape =  (16, 112, 112, 3)
            inputs = np.expand_dims(inputs, axis=0)  # input_shape =  (1, 16, 112, 112, 3)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))  # input_shape =  (1, 3, 16, 112, 112)
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()
            startTime = time.time()
            with torch.no_grad():
                outputs = model.forward(inputs)
            endTime = time.time() - startTime
            # print(endTime)
            outputs = np.squeeze(torch.sigmoid(outputs).cpu().numpy())

            output_frame = cv2.resize(frame, dsize=(640, 480))

            if (outputs > 0.5):
                prob = outputs
                cv2.putText(output_frame, "FallDown(prob: %.4f)" % prob, (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1.2,
                            (0, 0, 255), 2)
            else:
                prob = 1.0 - outputs
                cv2.putText(output_frame, "None(prob: %.4f)" % prob, (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1.2,
                            (255, 0, 0), 2)

            clip.pop(0)
            cv2.imshow('result', output_frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()