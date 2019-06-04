import torch
import dataloader
import network
from torch.utils.data import DataLoader
from torch import optim
from dataloader import UCF_VideoDataset
import numpy as np
import matplotlib.pyplot as plt
if __name__=="__main__":
    folder_path = 'D:/Dataset/UCF-101'
    readfile = 'D:/DATASET/UCF-101_recognition'
    # train_dataset = UCF_VideoDataset(folder_path=folder_path, readfile=readfile+'/trainlist01.txt')
    # test_dataset = UCF_VideoDataset(folder_path=folder_path, readfile=readfile+'/testlist01.txt')
    dataset = {'train': UCF_VideoDataset(folder_path=folder_path, readfile=readfile+'/trainlist01.txt'),
               'val': UCF_VideoDataset(folder_path=folder_path, readfile=readfile+'/testlist01.txt')}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
    batch_size = 8

    # train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_dataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloader = {x: DataLoader(dataset=dataset[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}

    my_model = network.C3DNet().cuda()
    # my_model.train()

    optimizer = optim.SGD(my_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    training_epoch = 100
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []

    for epoch in range(training_epoch):
        x_line = [i + 1 for i in range(epoch + 1)]
        epoch_loss = []
        loss = 0
        correct = 0
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            for it, data in enumerate(dataloader[phase]):
                if phase == 'train':
                    my_model.train()
                else:
                    my_model.eval()
                x = data[0].cuda()
                y = data[1].cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = my_model(x).cuda()
                    loss = criterion(y_pred, y.long())

                    _, y_pred = torch.max(y_pred, dim=1)
                    y_pred = y_pred.int()
                    correct += (y_pred == y).sum().item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(y_pred == y.data)
                print(running_loss, running_corrects)
                # if it % 100 == 0:
                #     _, y_pred = torch.max(y_pred, dim=1)
                #     y_pred = y_pred.int()
                #     correct += (y_pred == y).sum().item()
                #     print(correct / batch_size * 100)
                #     correct = 0
                # print("Iteration [{0}/{1}] Loss : {2:2.4f}".format(it, len(train_dataLoader), loss/batch_size))
                epoch_loss.append(loss.item()/batch_size)
            if phase == 'train':
                train_loss.append(np.mean(epoch_loss))
                plt.plot(x_line, train_loss, 'r-', label='train')
            else:
                val_loss.append(np.mean(epoch_loss))
                plt.plot(x_line, val_loss, 'b-', label='val')
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            '''
            epoch_loss = []
            correct = 0
            for it, data in enumerate(test_dataLoader):
                x = data[0].cuda()
                y = data[1].cuda()
                y_pred = my_model(x).cuda()

                loss = criterion(y_pred, y.long())
                _, y_pred = torch.max(y_pred, dim=1)
                y_pred = y_pred.int()
                correct += (y_pred == y).sum().item()

                epoch_loss.append(loss.item() / batch_size)
            val_loss.append(np.mean(epoch_loss))
            torch.save(my_model.state_dict(), 'Weights/C3D_{}.pt'.format(epoch + 1))
            '''
            print("Epoch : {0}/{1} Train_Loss : {2:2.4f} Val_Loss : {3:2.4f} Accuracy : {4:2.4f}"
                  .format(epoch+1, training_epoch, np.mean(train_loss), np.mean(val_loss), correct / dataset_sizes[phase] * 100))

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Convolution 3D Network')
            # plt.show()
            plt.savefig('Convolution 3D Network.png', dpi=300)
            torch.save(my_model.state_dict(), 'Weights/C3D_{}.pt'.format(epoch + 1))