import argparse
import os
import time

import cv2
from keras.losses import MeanSquaredError
from tensorflow import reduce_sum, square,reduce_mean
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from torch.autograd import Variable
from torchvision.utils import make_grid,save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

from models.DN_network import DN_Net, Discriminator
from utils.data_loader import ImageDataset, ImageTransform,make_data_path_list

torch.manual_seed(44)   #设置CPU生成随机数的种子
os.environ["CUDA_VISIBLE_DEVICES"]="0"  #设置GPU编号

#打印网络参数,统计模型的参数量有多少
def print_networks(net):
    num_params=0
    for param in net.parameters():
        num_params+=param.numel()
    print("Total number of parameters : %.3f M" % (num_params/1e6))

#获取超参数
def get_parser():
    parser=argparse.ArgumentParser(
        prog="D-N Net",  #程序的名称
        usage="python3 main.py",    #程序的用途
        description="This module demonstrates dehaze using D-N Net.",
        add_help=True   #为解析器添加一个-h/--help选项
    )
    #type-命令行参数应当被转换的类型；default-当参数未在命令行出现使用的值；help-一个此选项作用的简单描述
    parser.add_argument("-e","--epoch",type=int,default=1000,help="Number of epochs")
    parser.add_argument("-b","--batch_size",type=int,default=2,help="Batch size")
    parser.add_argument("-l","--load",type=str,default=None,help="The number of chechpoints")
    parser.add_argument("-hor","--hold_out_ratio",type=float,default=0.993,help="Training-Validation ratio")
    parser.add_argument("-s","--image_size",type=int,default=286)
    parser.add_argument("-cs","--crop_size",type=int,default=256)
    parser.add_argument("-lr","--lr",type=float,default=2e-4,help="Learning rate")

    return parser


def demo1(img):

    # 设置雾的密度和透射率
    fog_density = 0.8
    transmission = 0.5

    # 计算雾的值
    height, width, _ = img.shape
    fog = np.zeros((height, width), dtype=np.float32)
    fog[:, :] = fog_density

    # 计算透射率
    trans_map = np.exp(-transmission * fog)

    # 添加雾
    fog_img = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        fog_img[:, :, i] = img[:, :, i] * trans_map + fog * (1 - trans_map)

    # 将图像转换回uint8格式
    fog_img = np.uint8(fog_img)
    return fog_img

def fix_model_state_dict(state_dict):
    #初始化有序字典
    new_state_dict=OrderedDict()
    for k,v in state_dict.item:
        name=k
        if name.startswith("module."):
            name=name[7:]
        new_state_dict[name]=v
    return new_state_dict

#检查目录
def check_dir():
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    if not os.path.exists("./result"):
        os.mkdir("./result")

def set_requires_grad(nets,requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad=requires_grad

#损失日志
def plot_log(data, save_model_name="model"):
    plt.cla()
    plt.plot(data["G"],label="G_loss")
    plt.plot(data["D"],label="D_loss")
    plt.plot(data["SG"],label="Single_Generator_loss")
    plt.plot(data["GENERAL"],label="General_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.savefig("./logs/"+ save_model_name +".png")

def un_normalize(x):
    x=x.transpose(1,3)  #转置
    #mean,std
    x=x*torch.Tensor((0.5,))+torch.Tensor((0.5,))  #torch.Tensor()复制类
    x=x.transpose(1,3)  #归一化转化
    return x


# 需要显示解码器出来的噪声特征，本体特征，gt，加雾的图像
def evaluate(g1,dataset,device,filename):
    img,gt=zip(*[dataset[i] for i in range(9)])
    img=torch.stack(img)
    #gt_shadow=torch.stack(gt_shadow)
    gt=torch.stack(gt)
    print(gt.shape)
    print(img.shape)

    with torch.no_grad():
        reconstruct_c,reconstruct_n,reconstruct_gt=g1.test(img.to(device),gt.to(device))
        grid_rec=make_grid(un_normalize(reconstruct_c.to(torch.device("cpu"))),nrow=3)
        print(grid_rec.shape)
        reconstruct_n=reconstruct_n.to(torch.device("cpu"))
        reconstruct_gt = reconstruct_gt.to(torch.device("cpu"))
        reconstruct_c=reconstruct_c.to(torch.device("cpu"))

    grid_removal=make_grid(
        torch.cat(
            (
                un_normalize(img),
                un_normalize(gt),
                un_normalize(reconstruct_n),
                un_normalize(reconstruct_c),
                un_normalize(reconstruct_gt)
            ),
            dim = 0,
        ),
        nrow=9
    )
    save_image(grid_rec,filename+"noise_removal_img.jpg")
    save_image(grid_removal,filename+"noise_removal_separation.jpg")

def compute_loss(denoised_image, gt_image):
    # 将图像转换为Lab颜色空间
    #denoised_lab = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2Lab)
    #gt_lab = cv2.cvtColor(gt_image, cv2.COLOR_BGR2Lab)

    # 提取亮度（L）通道
    # denoised_l = denoised_image[:, :, 0]
    # gt_l = gt_image[:, :, 0]

    # 计算亮度损失（均方误差）
    # l_loss = np.mean((denoised_l - gt_l) ** 2)
    l_loss=MeanSquaredError()(denoised_image[:, :, :, 0], gt_image[:, :, :, 0])

    # 提取对比度（ab）通道
    denoised_ab = denoised_image[:, :, 1:]
    gt_ab = gt_image[:, :, 1:]

    # 计算对比度损失（均方误差）
    # ab_loss = np.mean((denoised_ab - gt_ab) ** 2)
    ab_loss = MeanSquaredError()(denoised_image[:, :, :, 1:], gt_image[:, :, :, 1:])
    # 返回亮度损失和对比度损失
    return l_loss, ab_loss


def train_model(g1,d1,dataloader,val_dataset,num_epochs,parser,save_model_name="model"):
    #检查项目路径
    check_dir()

    device="cuda:0" if torch.cuda.is_available() else "cpu"

    f_tensor=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    g1.to(device)
    d1.to(device)

    print("device:{}".format(device))

    lr=parser.lr


    beta1,beta2=0.5,0.999

    #优化器
    #params:待优化参数；lr:学习率；betas:用于计算梯度和梯度平方的运行平均值系数，默认为（0.9，0.999）
    optimizer_g=torch.optim.Adam([{"params":g1.parameters()}],lr=lr,betas=(beta1,beta2))

    #包装优化器
    #学习率这一块

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g,"min",factor=0.6,verbose=True,
                                                         threshold=0.00001,min_lr=0.000000000001,patience=50)

    #鉴别器
    optimizer_d=torch.optim.Adam([{"params":d1.parameters()}],lr=lr,betas=(beta1,beta2))

    #损失
    criterion_gan=nn.BCEWithLogitsLoss().to(device)   #sigmoid+BCE
    #criterion_gan=nn.BCELoss().to(device)
    criterion_l1=nn.L1Loss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    criterion_bce=nn.BCEWithLogitsLoss().to(device)

    mini_batch_size=parser.batch_size
    num_train_img=len(dataloader.dataset)
    batch_size=dataloader.batch_size

    lambda_dict={"lambda1":10,"lambda2":0.1,"lambda3":0.2}

    iteration=1
    g_losses=[]
    d_losses=[]
    general_losses=[]
    single_gan_losses=[]

    start=0
    if parser.load is not None:
        start=int(parser.load)+1
    for epoch in range(start,num_epochs+1):
        g1.train()
        d1.train()

        t_epoch_start=time.time()

        epoch_g_loss=0.0
        epoch_d_loss=0.0
        epoch_single_g_loss=0.0
        epoch_tf_loss=0.0

        print('--------------')
        print('Epoch  {}/{}'.format(epoch,num_epochs))
        print('(train)')

        data_len=len(dataloader)
        print("data_len={}".format(data_len))

        for images,gt in tqdm(dataloader):
            #默认加载两张图片，batch_size==1时可能会出现错误
            if images.size()[0]==1:
                continue
            images=images.to(device)
            gt=gt.to(device)

            mini_batch_size=images.size()[0]

            #====训练鉴别器=====

            #允许反向传播
            set_requires_grad([d1],True)
            #将模型参数梯度设置为0
            optimizer_d.zero_grad()
            #获取生成器生成的图片
            reconstruct_clean,reconstruct_no,reconstruct_gt=g1(images,gt)
            #计算亮度损失l_loss,和对比度损失ab_loss
            l_loss=criterion_mse(reconstruct_gt[:, :, :, 0],gt[:, :, :, 0])
            ab_loss=criterion_mse(reconstruct_gt[:, :, :, 1:],gt[:, :, :, 1:])

            fake1=torch.cat([images,reconstruct_clean],dim=1)#输入图片和生成噪声图片cat连接
            real1=torch.cat([images,gt],dim=1)#将输入图片和gt做cat连接

            out_d1_fake=d1(fake1.detach())  #detach()截断反向传播流
            out_d1_real=d1(real1) #

            label_d1_fake=Variable(f_tensor(np.zeros(out_d1_fake.size())),requires_grad=True)
            label_d1_real=Variable(f_tensor(np.ones(out_d1_fake.size())),requires_grad=True)

            #计算损失
            loss_d1_fake=criterion_gan(out_d1_fake,label_d1_fake)
            loss_d1_real=criterion_gan(out_d1_real,label_d1_real)

            #鉴别器/判别器。
            # 判别器使用真实图像和生成器生成的图像进行训练，以便学习区分真实图像和生成图像的能力
            d_loss=lambda_dict["lambda2"]*loss_d1_fake+lambda_dict["lambda2"]*loss_d1_real
            d_loss.backward()
            optimizer_d.step()  #对所有参数进行更新
            epoch_d_loss+=d_loss.item()

            #=====训练生成器======
            #生成器生成的图像计算对抗损失函数和重构损失函数，并将它们相加得到总的损失函数
            #根据中损失函数对生成器进行反向传播，更新生成器的参数。
            set_requires_grad([d1],False)
            optimizer_g.zero_grad()

            #使用鉴别器帮助生成器训练
            fake1=torch.cat([images,reconstruct_clean],dim=1) #输入图片和生成无阴影图片cat连接
            out_d1_fake=d1(fake1.detach())
            g_l_c_gan1=criterion_gan(out_d1_fake,label_d1_real)

            # 计算噪声图像和干净图像的差异
            diff = torch.abs(images - gt)
            # 根据差异计算掩码
            threshold = 0.05  # 可以根据具体情况调整阈值
            mask = (diff > threshold).float()  # 大于阈值的位置被置为 1，否则为 0

            #分别计算  输入重构损失  无噪声重构损失  gt重构损失
            g_l_data1=criterion_bce(reconstruct_no,mask)  #噪声特征图像损失
            g_l_data2=criterion_mse(reconstruct_clean,gt) #无噪声重构损失
            g_l_data3=criterion_mse(reconstruct_gt,gt)  #gt重构损失

            #生成器总损失
            g_loss=lambda_dict["lambda1"]*g_l_data1+g_l_data2+\
                lambda_dict["lambda1"]*g_l_data3+lambda_dict["lambda2"]*g_l_c_gan1+lambda_dict["lambda2"]*(l_loss+ab_loss)

            g_loss.backward()
            optimizer_g.step()
            # print(g_l_data4)

            epoch_g_loss+=g_loss.item()    #生成器总损失
            epoch_single_g_loss+=g_l_c_gan1.item()  #gan损失
            epoch_tf_loss+=g_l_data2.item() #无噪声重构损失

        t_epoch_finish=time.time()
        Epoch_D_Loss=epoch_d_loss/(lambda_dict["lambda2"]*2*data_len)
        Epoch_G_Loss=epoch_g_loss/data_len
        Epoch_Single_G_Loss=epoch_single_g_loss/data_len
        Epoch_tf_Loss=epoch_tf_loss/data_len

        print("------------")
        print("epoch {}  || Epoch_D_Loss:{:.4f}  || Epoch_G_Loss:{:.4f} ||  Epoch_Single_G_Loss:{:.4f}  ||  Epoch_tf_Loss:{:.4f}".format(
            epoch,
            epoch_d_loss/(lambda_dict["lambda2"]*2*data_len),
            epoch_g_loss/data_len,
            epoch_single_g_loss/data_len,
            epoch_tf_loss/data_len,
        ))
        print("timer:{:.4f} sec.".format(t_epoch_finish-t_epoch_start))

        #d_losses+=[epoch_d_loss/(lambda_dict["lambda2"]*2*data_len)]
        #g_losses+=[epoch_g_loss/data_len]
        scheduler.step(epoch_g_loss/data_len)

        t_epoch_start=time.time()

        # g_losses=np.append(epoch_d_loss/(lambda_dict["lambda2"]*2*data_len))
        # d_losses=np.append(epoch_g_loss/data_len)
        # single_gan_losses=np.append(epoch_single_g_loss/data_len)
        # general_losses=np.append(epoch_tf_loss/data_len)
        g_losses.append(Epoch_D_Loss)
        d_losses.append(Epoch_G_Loss)
        single_gan_losses.append(Epoch_Single_G_Loss)
        general_losses.append(Epoch_tf_Loss)

        #输出损失日志
        #plot_log(
         #  {
         #      "G":g_losses,
          #     "D":d_losses,
          #     "SG":single_gan_losses,
         #      "GENERAL":general_losses
        #   },
        #   save_model_name+str(epoch)
       # )

        #采用间隔几个epoch保存模型
        if epoch%10==0:
            torch.save(g1.state_dict(),"checkpoints/"+save_model_name+"_G1_"+str(epoch)+".pth")
            torch.save(d1.state_dict(),"checkpoints/"+save_model_name+"_D1_"+str(epoch)+".pth")

            g1.eval()
            evaluate(g1,val_dataset,device,"{:s}/val_{:d}".format("result",epoch))

    return g1

#模型训练
def train(parser):
    #初始化生成器和鉴别器
    g1=DN_Net(input_channels=3,output_channels=3)
    d1=Discriminator(input_channels=6)

    print_networks(g1)
    print_networks(d1)

    #是否加载已有模型
    if parser.load is not None:
        print("load checkpoint"+parser.load)
        g1.load_state_dict(fix_model_state_dict(torch.load("./checkpoints/S-R-Net_G1_"+parser.load+'.pth')))
        d1.load_state_dict(fix_model_state_dict(torch.load("./checkpoints/S-R-Net_D1_"+parser.load+'.pth')))

    #取出训练集和验证集路径
    #train_img_list,val_img_list=make_data_path_list(phase='train',rate=parser.hold_out_ratio)[:20]
    train_img_list, val_img_list = make_data_path_list(phase='train',rate=0.95)[:20]
    #print(len(train_img_list["path_A"]))
    #print(len(val_img_list['path_A']))
    print("train_dataset:{}".format(len(train_img_list["path_A"])))
    print("val_dataset:{}".format(len(val_img_list['path_A'])))

    mean=(0.5,)
    std=(0.5,)
    size=parser.image_size
    crop_size=parser.crop_size
    batch_size=parser.batch_size
    num_epochs=parser.epoch

    #数据加载器+预处理
    train_dataset=ImageDataset(img_list=train_img_list,
                               img_transform=ImageTransform(size=size,crop_size=crop_size,mean=mean,std=std),
                               phase="train")
    val_dataset=ImageDataset(img_list=val_img_list,
                             img_transform=ImageTransform(size=size,crop_size=crop_size,mean=mean,std=std),
                             phase='test_no_crop')
    # train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=6)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False, num_workers=6)


    g1=train_model(g1,d1,dataloader=train_dataloader,
                   val_dataset=val_dataset,
                   num_epochs=num_epochs,
                   parser=parser,
                   save_model_name="D-N-Net")

if __name__=="__main__":
    m_parser=get_parser().parse_args()
    train(m_parser)