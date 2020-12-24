import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import  glob
import  random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import  DataLoader
import torchvision.transforms as transforms

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 224
crop_sieze = 800



class BatchRename():
    def __init__(self):
        self.path = '.\machineLearning\Train3\Train1\\'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1001
        for item in filelist:
            if item.endswith('.jpeg') or item.endswith('.png') or item.endswith('.gif') or item.endswith('.bmp') or item.endswith('.webp'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), item[:item.rfind(".")]+".jpg")
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

class CustomDataset(Dataset):
    def __init__(self, path,label_file_path):
        # with open(label_file_path, 'r') as f:
        #     # (image_path(str), image_label(str))
        #     self.imgs = list(map(lambda line: line.strip().split(' '), f))
        my_label = pd.read_csv(label_file_path)
        id = np.array(my_label['FileName'])
        type = np.array(my_label['type'])
        mp = {}
        for i in range(len(id)):
            id[i] = id[i][:id[i].rfind(".")] + ".jpg"
        # print("id\n", id)
        for i in range(len(id)):
            mp[id[i]] = type[i]
        imglists = []
        image_paths = list(glob.glob(path))
        image_paths = [str(path) for path in image_paths]  # 所有图片路径的列表
        random.shuffle(image_paths)  # 打散
        for path in image_paths:
            # print("path\n",path)
            string2 = path[path.rfind("\\"):]  # 在strint1中查找最后一个正斜杠/后面的字符,图片名称
            imglist = string2[1:]
            imglist = imglist[:imglist.rfind(".")]
            imglist = imglist + ".jpg"
            imglists.append(imglist)

            # print(imglist)
        image_labels = np.array(my_label['type'])
        labels = []
        mp2={}
        imglists1 = []
        tmp=[]
        for imglist in imglists:
            #print("imglist\n",imglist)
            key = mp.__contains__(imglist)
            if (key):
                labels.append(mp[imglist])
                # imglists1.append(".\machineLearning\Train3\Train1\\" + imglist)
                # img=".\machineLearning\Train3\Train1\\" + imglist

                imglists1.append(".\machineLearning\Train4\Train1\\" + imglist)
                img=".\machineLearning\Train4\Train1\\"+ imglist
                # imglists1.append(imglist)
                # img=imglist

                #self.imgs = list(mp("G:\machineLearning\Train3\Train1\\" + imglist,mp[imglist]))
            tmp.append([img,mp[imglist]])
        self.imgs=tmp
        # print("imgs\n",self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        # print("path\n",path)
        # print("label\n",label)
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_sieze, crop_sieze), pad_if_needed=True),

                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(0.25, 0.25),
                transforms.ToTensor(),
                # transforms.ToTensor(),
                # transforms.Resize((224,224)),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img = transform(image)
        #img = transforms.Compose([transforms.ToTensor()])
        label = int(label)
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

class BatchRename1():
    def __init__(self):
        self.path = '.\machineLearning\Test1\Test\\'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1001
        for item in filelist:
            if item.endswith('.jpeg') or item.endswith('.png') or item.endswith('.gif') or item.endswith('.bmp') or item.endswith('.webp'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), item[:item.rfind(".")]+".jpg")
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

class CustomDataset1(Dataset):
    def __init__(self, path):
        # with open(label_file_path, 'r') as f:
        #     # (image_path(str), image_label(str))
        #     self.imgs = list(map(lambda line: line.strip().split(' '), f))
        imglists = []
        image_paths = list(glob.glob(path))
        image_paths = [str(path) for path in image_paths]  # 所有图片路径的列表
        random.shuffle(image_paths)  # 打散
        for path in image_paths:
            # print("path\n",path)
            string2 = path[path.rfind("\\"):]  # 在strint1中查找最后一个正斜杠/后面的字符,图片名称
            imglist = string2[1:]
            imglist = imglist[:imglist.rfind(".")]
            imglist = imglist + ".jpg"
            imglists.append(imglist)


        imglists1 = []
        tmp=[]
        for imglist in imglists:
                img=".\machineLearning\Test1\Test\\" + imglist

                #self.imgs = list(mp("G:\machineLearning\Train3\Train1\\" + imglist,mp[imglist]))
                tmp.append([img,0])
        self.imgs=tmp
        # print("imgs\n",self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        # print("path\n",path)
        # print("label\n",label)
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform(image)
        #img = transforms.Compose([transforms.ToTensor()])
        return img

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 224, 224)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=64,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 224, 224)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=1),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)  (16, 112, 64)
            nn.Conv2d(64, 128, 3, 1, 1),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)  (32,64,32)
        )
        self.conv3 = nn.Sequential(  # input shape (3, 224, 128)
            nn.Conv2d(128,256,3,1,1),  # output shape (16, 256, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv4 = nn.Sequential(  # input shape (3, 224, 128)
            nn.Conv2d(256, 512, 3, 1, 1),  # output shape (16, 256, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv5 = nn.Sequential(  # input shape (3, 224, 128)
            nn.Conv2d(512, 512, 3, 1, 1),  # output shape (16, 256, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv6 = nn.Sequential(  # input shape (3, 224, 128)
            nn.Conv2d(512, 512, 3, 1, 1),  # output shape (16, 256, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.fc1=nn.Linear(512 * 7 * 7, 4096)   # fully connected layer, output 10 classes
        self.fc2=nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x=self.fc1(x)
        x=self.fc2(x)
        output = self.out(x)
        return output

if __name__ == '__main__':

    # rename=BatchRename()
    # rename.rename()
    # rename=BatchRename1()
    # rename.rename()
    # test_path='.\machineLearning\Test1\Test\*'
    # train_path = '.\machineLearning\Train3\Train1\*'
    train_path = '.\machineLearning\Train4\Train1\*'
    csv_path = '.\machineLearning\Train_label.csv'
    train_data = CustomDataset(train_path,csv_path)
    train_size=0.8*len(train_data)
    train_dataset=[]
    test_dataset=[]
    train_size=int(train_size)
    print("train_size\n",train_size)
    for i in range(train_size):
        train_dataset.append(train_data[i])
    for i in range(train_size, len(train_data)):
        test_dataset.append(train_data[i])
    # test_data= CustomDataset1(test_path)
    # dataloaders1 = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    # for i, item in enumerate(train_data):
    #     data, label = item
    #     print('data:', data)
    #     print('label:', label)
    EPOCH = 50  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 50
    LR = 0.0001





    dataloaders = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    cnn = CNN().cuda(device=device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # for step, (b_x, b_y) in enumerate(dataloaders):
    #     print("b_x\n",b_x)
    #     print("b_y\n", b_y)

    # training and testing
    loss_y=[]
    acc_y=[]
    for epoch in range(EPOCH):
        train_acc = 0
        i=0
        for step, (b_x, b_y) in enumerate(dataloaders):  # 分配 batch data, normalize x when iterate train_loader
            i=i+1
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #with torch.no_grad():
            output = cnn(b_x)  # cnn output
            # print("output: ",output)
            loss = loss_func(output, b_y)  # cross entropy loss
            pred_y = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
            print("pred_y:", pred_y)
            b_y = torch.unsqueeze(b_y, dim=0)
            pred_y = torch.unsqueeze(torch.tensor(pred_y), dim=0).cuda()
            train_correct = (pred_y == b_y.squeeze(1)).sum().cuda()
            train_acc += int(train_correct)
            print("pred_y ", pred_y)
            print("b_y ", b_y)

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #print("loss\n", loss)
            #print(int(train_correct), len(b_y))
        loss_y.append(loss)
        print("准确率\n", 1.0 * train_acc / (EPOCH * BATCH_SIZE))
        acc_y.append(1.0 * train_acc / (EPOCH * BATCH_SIZE))

    x=np.linspace(1,EPOCH,EPOCH)
    plt.plot(x,loss_y)
    plt.title("损失")
    plt.show()
    plt.plot(x, acc_y)
    plt.title("准确率")
    plt.show()

    #torch.save(cnn, "cnn4.pth")
    #cnn = torch.load("cnn1.pth")
    i=0
    train_acc=0
    cnn=cnn.eval()
    dataloaders = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)
    for step, (b_x,b_y) in enumerate(dataloaders):
        i=i+1
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        test_output = cnn(torch.tensor(b_x))
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        b_y = torch.unsqueeze(b_y, dim=0)
        pred_y = torch.unsqueeze(torch.tensor(pred_y), dim=0).cuda()
        train_correct = (pred_y == b_y.squeeze(1)).sum().cuda()
        train_acc += int(train_correct)
        print('第%i '%i )
        print(" ",pred_y," ",b_y)


    print("准确率\n",1.0*train_acc / (len(test_dataset)))