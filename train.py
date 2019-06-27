# -*- coding: utf-8 -*-
"""
@author: Administrator
"""
from torch.utils import data
import os
from PIL import  Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from data_pre import create_test  
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models
from tqdm import tqdm 
warnings.filterwarnings("ignore")

#   to the ImageFolder structure
data_dir = "/media/dell/dell/data/遥感/"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 45

# Batch size for training (change depending on how much memory you have)
batch_size = 128     #批处理尺寸(batch_size)

# Number of epochs to train for 
EPOCH = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
#feature_extract = True
feature_extract =False
# 超参数设置
pre_epoch = 0  # 定义已经遍历数据集的次数



from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return t.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    net = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        net = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        net = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        net = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        net = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        net.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        net = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, num_classes)
        #pre='/home/dell/Desktop/zhou/train3/net_024.pth'
        #net.load_state_dict(torch.load(pre)) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        net = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # Handle the auxilary net
        num_ftrs = net.AuxLogits.fc.in_features
        net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return net, input_size


# Initialize the model for this run
net, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(net, input_size)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#训练使用多GPU，测试单GPU
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

net.to(device)
# Send the model to GPU
net = net.to(device)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#print(len(image_datasets['train']))
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=16,collate_fn=my_collate_fn) for x in ['train', 'val']}
net.class_to_idx = image_datasets['train'].class_to_idx
net.idx_to_class = {
    idx: class_
    for class_, idx in net.class_to_idx.items()}
c={0: 26, 1: 30, 2: 13, 3: 31, 4: 40, 5: 6, 6: 16, 7: 11, 8: 23, 9: 41, 10: 24, 11: 9, 12: 15, 13: 1, 14: 28, 15: 5, 16: 29, 17: 3, 18: 17, 19: 2, 20: 39, 21: 25, 22: 7, 23: 8, 24: 44, 25: 21, 26: 35, 27: 14, 28: 45, 29: 37, 30: 27, 31: 42, 32: 10, 33: 43, 34: 34, 35: 18, 36: 22, 37: 4, 38: 38, 39: 20, 40: 32, 41: 33, 42: 36, 43: 12, 44: 19}
#print(list(net.idx_to_class.items()))
test_files = pd.read_csv("/home/dell/Desktop/1.csv")
test_gen = create_test(test_files,'/media/dell/dell/data/遥感/test/',augument=False,mode="test")
test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=16)
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch densenet Training')
parser.add_argument('--outf', default='/home/dell/Desktop/zhou/train3/', help='folder to output images and model checkpoints') #输出结果保存路径
#parser.add_argument('--net', default='/home/dell/Desktop/zhou/resnet.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()
params_to_update = net.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model):
    sample_submission_df = pd.read_csv("/home/dell/Desktop/1.csv")
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.to(device)
    model.eval()
    submit_results = []
    for i,(input,filepath) in tqdm(enumerate(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        #print(filepath)
        #print(input)
        with torch.no_grad():
            image_var = input.to(device)
            y_pred = model(image_var)
#            y_pred = model(image_var,visit)
            label=y_pred.cpu().data.numpy()
            labels.append(label==np.max(label))
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow=np.argmax(row)
        subrow=c[subrow]
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('/home/dell/Desktop/densent.csv', index=None)


def main():
    ii=0
    LR = 0.001        #学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, densenet!")  # 定义遍历数据集的次数
    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(params_to_update, lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    #optimizer = optim.SGD(params_to_update, lr=LR, momentum=0.9) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    with open("/home/dell/Desktop/zhou/train3/acc.txt", "w") as f:
        with open("/home/dell/Desktop/zhou/train3/log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(dataloaders_dict['train'], 0):
                    # 准备数据
                    length = len(dataloaders_dict['train'])
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 训练
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in dataloaders_dict['val']:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).cpu().sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    
                    # 将每次测试结果实时写入acc.txt文件中
                    if (ii%1==0):
                        print('Saving model......')
                        torch.save(net.module.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("/home/dell/Desktop/zhou/train3/best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
if __name__ == "__main__":
    main()
#    test(test_loader,net)

'''

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


def visualize_model(net, num_images=10):
    net.eval()
    images_so_far = 0
 
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
 
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[predicted[j]]))
            imshow(inputs.cpu().data[j])
            if images_so_far == num_images:
                return
visualize_model(net)       #显示十张图片
plt.show()

'''














