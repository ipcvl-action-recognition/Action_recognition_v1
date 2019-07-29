import torch
from torch.autograd import Variable
import dataloader
import network
from torch.utils.data import DataLoader
from torch import optim
from dataloader import Le2i_VideoDataset
from dataloader import UCF101_Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import cv2
import visdom
import sys
from networks import R2Plus1D_model
from torch.optim.lr_scheduler import StepLR
from test import test_accuracy

if __name__ == "__main__":
    folder_path = './Test_data/Video'
    readfile = './Test_data/list'
    framefile = './Test_data/frame_label'



    train_dataset = Le2i_VideoDataset(folder_path=folder_path + '/Train', readfile=readfile + '/trainlist.txt',
                                      framefile=framefile + '/Train')
    val_dataset = Le2i_VideoDataset(folder_path=folder_path + '/Val', readfile=readfile + '/vallist.txt',
                                    framefile=framefile + '/Val', is_train=False)

    batch_size = 5
    lr = 1e-6

    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
                                  pin_memory=False)
    val_dataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
                                pin_memory=False)
    my_model = network.C3DNet(pretrained=True).cuda()
    #my_model2 = R2Plus1D_model.R2Plus1DClassifier(101, (2, 2, 2, 2), pretrained=False)
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion = torch.nn.BCELoss().cuda()

    # train_params = [{'params': network.get_1x_lr_params(my_model), 'lr': lr},
    #                {'params': network.get_10x_lr_params(my_model), 'lr': lr}]

    # optimizer = optim.Adam(train_params)
    optimizer = optim.Adam(my_model.parameters(), lr=lr, weight_decay=lr/2)
    # optimizer = optim.Adam(my_model.parameters(), lr=lr)

    training_epoch = 10000

    train_loss = []
    val_loss = []
    accuracy_list = []

    vis = visdom.Visdom()

    loss_window = vis.line(X=torch.zeros((1, 2)).cpu(),
                           Y=torch.zeros((1, 2)).cpu(),
                           opts=dict(xlabel='Epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Train Loss', 'Val Loss']))

    acc_window = vis.line(X=torch.zeros((1)).cpu(),
                          Y=torch.zeros((1)).cpu(),
                          opts=dict(xlabel='Epoch',
                                    ylabel='Acc',
                                    title='Validation  Accuracy',
                                    legend=['Val Acc']))
    best_acc = sys.float_info.min

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(training_epoch + 1):
        train_epoch_losses = []

        accuracy = 0
        correct = 0

        scheduler.step()
        my_model.train()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        isPut = True

        for it, data in enumerate(train_dataLoader):
            x = data[0].cuda()  # buffer
            y = data[1].cuda()  # label - one_hot

            optimizer.zero_grad()

            logits = my_model(x).cuda()

            y_pred = torch.sigmoid(logits)

            loss = criterion(y_pred.squeeze(), y.float())

            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                print("epoch {0} Iteration [{1}/{2}] Train_Loss : {3:2.4f}".format(epoch, it, len(train_dataLoader),
                                                                                   loss))
                print("y_pred = ", torch.transpose(y_pred, 0, 1))
                print("y = ", y)
                print("logits = ", torch.transpose(logits, 0, 1))
                print()
            train_epoch_losses.append(loss.item())

            if isPut:
                _x = x[0, :]
                _x = _x.permute(1, 0, 2, 3)
                vis.images(_x, win="Img")
                isPut = False
            del loss
            del logits

        val_epoch_losses = []
        my_model.eval()
        total_len = 0
        for it, data in enumerate(val_dataLoader):
            with torch.no_grad():
                val_epoch_losses, acc = test_accuracy(is_validation=True, video_list=data, folder_path=framefile)



                '''
                while frame_num+16 < video_len[batch]:
                    inputs = x[batch, :, frame_num:frame_num+16, :, :]
                    inputs = torch.unsqueeze(inputs, 0)
                    # print(frame_num, inputs.shape, float(y[batch][frame_num]))
                    inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()
                    logits = my_model(inputs).cuda()
                    y_pred = torch.sigmoid(logits)
                    # print(y_pred.squeeze(0).shape, np.array([float(y[batch][frame_num+16])]).shape)
                    y_frame = torch.from_numpy(np.array([float(y[batch][frame_num+16])])).cuda()
                    loss = criterion(y_pred.squeeze(0), y_frame.float()).cuda()
                    y_pred_index = torch.round(y_pred).int()
                    # print(y_pred_index)
                    y_pred_index = torch.transpose(y_pred_index, 0, 1)
                    y_frame = y_frame.int()

                    correct += (y_pred_index == y_frame).sum().item()
                    # print(loss.item())
                    val_epoch_losses.append(loss.item())
                    frame_num += 1
                    total_len += 1
                '''
                '''
                logits = my_model(x).cuda()
                y_pred = torch.sigmoid(logits)
                loss = criterion(y_pred.squeeze(), y.float())

                y_pred_index = torch.round(y_pred).int()
                y_pred_index = torch.transpose(y_pred_index, 0, 1)
                y = y.int()

                correct += (y_pred_index == y).sum().item()
                val_epoch_losses.append(loss.item())
                '''
        mean_epoch_train_loss = np.mean(train_epoch_losses)
        mean_epoch_val_loss = np.mean(val_epoch_losses)

        vis.line(X=np.column_stack((np.array([epoch]), np.array([epoch]))),
                 Y=np.column_stack((np.array([mean_epoch_train_loss]), np.array([mean_epoch_val_loss]))),
                 win=loss_window, update='append')


        # acc = 100 * (correct / val_dataset.__len__())
        vis.line(X=np.array([epoch]), Y=np.array([acc]), win=acc_window, update='append')

        print("epoch {0} Train_mean_Loss : {1:2.4f}  val_mean_Loss : {2:2.4f}".format(epoch, mean_epoch_train_loss,
                                                                                      mean_epoch_val_loss))
        print("val_accuracy : ", acc)
        print()
        print()

        if best_acc < acc or epoch+1 % 50 == 0:
            print("Saved New Weight {0:2.2f} to {1:2.2f} acc".format(best_acc, acc))
            best_acc = acc
            torch.save(my_model.state_dict(),
                       'Weights/Falldown_{0}_{1:2.2f}_{2:2.2f}_{3:2.2f}.pt'.format(epoch + 1, mean_epoch_train_loss,
                                                                                   mean_epoch_val_loss, best_acc))
            print()
            print()

        else:
            print("There are no improve than {0:2.2f} acc".format(best_acc))