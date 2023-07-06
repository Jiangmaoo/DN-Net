import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from torch.autograd import Variable
from torchvision.utils import make_grid,save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

from models.DN_network import DN_Net, Discriminator
from models.MeanFilter import MeanFilter
from utils.data_loader import ImageDataset, ImageTransform,make_data_path_list

torch.manual_seed(44)   #设置CPU生成随机数的种子
operation_seed_counter = 0
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

#返回一个CUDA设备上的生成器
def get_generator(dev):
    global operation_seed_counter  #定义一个全局变量，用于生成操作的种子
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=dev)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

#用于将输入张量x进行空间到深度的转换，再一定程度上增加网络的非线性容量，处进行信息的流动和特征的重用
def space_to_depth(x, block_size):
    n, c, h, w = x.size()  #获取张量x的维度信息
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=1,padding=1)  #使用unfold函数对输入张量进行展开操作，将图像歌城block*block的块
    return unfolded_x.view(n, c * block_size**2, h, w)   #实现空间到深度的转换。3 1 1

#为给定图像生成一对掩码
def generate_mask_pair(img,dev):
    # prepare masks (N x C x H x W)
    n, c, h, w = img.shape
    #创建bool类型张量存储掩码
    mask1 = torch.zeros(size=(n * h * w * 9, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h * w * 9, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1], [2, 5], [3, 0], [3, 4], [3, 6], [4, 3], [4, 1],
         [4, 5], [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 6], [7, 4], [7, 8], [8, 7], [8, 5]],
        dtype=torch.int64,
        device=img.device)
    #创建张量存储随机生成的索引
    rd_idx = torch.zeros(size=(n * h * w, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h * w ,),
                  generator=get_generator(dev),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    #索引加上一定的偏移量
    rd_pair_idx += torch.arange(start=0,
                                end=n * h * w * 9,
                                step=9,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

#生成图像的子图像。根据给定的图像和掩码，生成图像的子图像。
# 其中子图像是将原图像按照2*2的块进行划分，并再每个块内选择掩模为 True的像素，生成对应的子图像，
# 这样可以将图像分解成多个更小的图像块。
def generate_subimages(img, mask):
    n, c, h, w = img.shape
    #创建一个与原图像相同大小的零张量，作为子图像的容器
    subimage = torch.zeros(n,c, h, w,dtype=img.dtype,layout=img.layout,device=img.device)
    # per channel，针对每个通道进行操作
    for i in range(c):
        #将当前通道的图像通过该函数进行空间到深度的转换，得到h/2×w/2的张量
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=3)
        #对张量进行维度变换，将通道维度移动到最后，并展平为一维
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        #根据掩模mask从img_中选择相应像素，并将调整维度顺序。
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h, w, 1).permute(0, 3, 1, 2)
    return subimage

def calNoise(images,dev):
    mask1, mask2 = generate_mask_pair(images,dev) #mask1.shape:torch.Size([65536]) 65536=256*256
    noisy_sub1 = generate_subimages(images, mask1)  #noisy_sub1.shape:torch.Size([1, 3, 128, 128])
    noisy_sub2 = generate_subimages(images, mask2)

    mean_filter = MeanFilter(kernel_size=5)
    denoise1 = mean_filter(noisy_sub1) #torch.Size([1, 3, 128, 128])
    denoise2 = mean_filter(noisy_sub2)

    # 计算噪声图像和干净图像的差异
    diff1 = torch.abs(images - denoise1)
    # 根据差异计算掩码
    threshold = 0.05  # 可以根据具体情况调整阈值
    noise1 = (diff1 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0
    diff2 = torch.abs(images - denoise2)
    noise2 = (diff2 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0

    return noise1,noise2


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
    plt.figure()
    ax1 = plt.subplot(221)
    ax1.plot(data["G"])
    ax1.set_title("G_loss")

    ax2 = plt.subplot(222)
    ax2.plot(data["D"])
    ax2.set_title("D_loss")

    ax3 = plt.subplot(223)
    ax3.plot(data["SG"])
    ax3.set_title("Single_Generator_loss")

    ax4 = plt.subplot(224)
    ax4.plot(data["GENERAL"])
    ax4.set_title("General_loss")

    plt.tight_layout()
    # ax1=plt.subplot(221)
    # ax1.plot(data["G"],label="G_loss")
    # ax2 = plt.subplot(222)
    # ax2.plot(data["D"],label="D_loss")
    # ax3 = plt.subplot(223)
    # ax3.plot(data["SG"],label="Single_Generator_loss")
    # ax4 = plt.subplot(224)
    # ax4.plot(data["GENERAL"],label="General_loss")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("Loss")
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
    # print(img.device)  #cpu

    noise1, noise2 = calNoise(img,"cpu")
    with torch.no_grad():

        noise_ave,noise_no,reconstruct_gt=g1.test(img.to(device),noise1.to(device),noise2.to(device))
        grid_rec=make_grid(un_normalize(reconstruct_gt.to(torch.device("cpu"))),nrow=3)
        # print(grid_rec.shape)
        noise_ave=noise_ave.to(torch.device("cpu"))
        reconstruct_gt = reconstruct_gt.to(torch.device("cpu"))
        noise_no=noise_no.to(torch.device("cpu"))

    grid_removal=make_grid(
        torch.cat(
            (
                un_normalize(img),
                un_normalize(gt),
                un_normalize(reconstruct_gt),
                un_normalize(noise_ave),
                un_normalize(noise_no)

            ),
            dim = 0,
        ),
        nrow=9
    )
    save_image(grid_rec,filename+"noise_removal_img.jpg")
    save_image(grid_removal,filename+"noise_removal_separation.jpg")


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
    optimizer_d1=torch.optim.Adam([{"params":d1.parameters()}],lr=lr,betas=(beta1,beta2))

    #损失
    criterion_gan=nn.BCEWithLogitsLoss().to(device)   #sigmoid+BCE
    #criterion_gan=nn.BCELoss().to(device)
    criterion_l1=nn.L1Loss().to(device)
    criterion_mse = nn.MSELoss().to(device) #均方差损失函数MSELoss L2

    criterion_bce=nn.BCEWithLogitsLoss().to(device)

    mini_batch_size=parser.batch_size
    num_train_img=len(dataloader.dataset)
    batch_size=dataloader.batch_size

    lambda_dict={"lambda1":10,"lambda2":0.1,"lambda3":0.2,"lambda4":2}

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
        epoch_nd_loss = 0.0
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

            # noise1,noise2=calNoise(images)
            mask1, mask2 = generate_mask_pair(images,device)  # mask1.shape:torch.Size([65536]) 65536=256*256
            noisy_sub1 = generate_subimages(images, mask1)  # noisy_sub1.shape:torch.Size([1, 3, 128, 128])
            noisy_sub2 = generate_subimages(images, mask2)

            mean_filter = MeanFilter(kernel_size=5)
            denoise1 = mean_filter(noisy_sub1)  # torch.Size([1, 3, 128, 128])
            denoise2 = mean_filter(noisy_sub2)

            # 计算噪声图像和干净图像的差异
            diff1 = torch.abs(images - denoise1)
            # 根据差异计算掩码
            threshold = 0.05  # 可以根据具体情况调整阈值
            noise1 = (diff1 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0
            diff2 = torch.abs(images - denoise2)
            noise2 = (diff2 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0

            #====训练鉴别器=====

            #允许反向传播
            set_requires_grad([d1],True)
            #将模型参数梯度设置为0
            optimizer_d1.zero_grad()
            #获取生成器生成的图片
            noise_ave, noise_no,reconstruct_gt=g1(images,noise1,noise2)


            #重建干净图像鉴别器
            fake1=torch.cat([images,reconstruct_gt],dim=1)#输入图片和生成噪声图片cat连接
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
            optimizer_d1.step()  #对所有参数进行更新
            epoch_d_loss+=d_loss.item()

            #=====训练生成器======
            #生成器生成的图像计算对抗损失函数和重构损失函数，并将它们相加得到总的损失函数
            #根据中损失函数对生成器进行反向传播，更新生成器的参数。
            set_requires_grad([d1],False)
            optimizer_g.zero_grad()

            #使用鉴别器帮助生成器训练
            fake1=torch.cat([images,reconstruct_gt],dim=1) #输入图片和生成无阴影图片cat连接
            out_d1_fake=d1(fake1.detach())
            g_l_c_gan1=criterion_gan(out_d1_fake,label_d1_real)

            # 计算噪声图像和干净图像的差异
            diff = torch.abs(images - gt)
            # 根据差异计算掩码
            threshold = 0.05  # 可以根据具体情况调整阈值
            mask = (diff > threshold).float()  # 大于阈值的位置被置为 1，否则为 0

            #分别计算
            g_l_data1=criterion_mse(reconstruct_gt,gt)  #gt重构损失
            g_l_data2 = criterion_mse(noise_ave, mask)  # 噪声分布图像损失
            g_l_data3 = criterion_mse(noise_no, gt)  # 减噪噪图像损失
            # g_l_data2=criterion_bce(noise1,noise2)     #噪声特征相似损失

            #生成器总损失
            # g_loss=lambda_dict["lambda1"]*g_l_data1+g_l_data2+\
            #     lambda_dict["lambda1"]*g_l_data3+lambda_dict["lambda2"]*g_l_c_gan1+lambda_dict["lambda2"]*(l_loss+ab_loss)

            # 生成器总损失10,0.1,0.2,2
            g_loss=lambda_dict["lambda1"]*g_l_data1+lambda_dict["lambda2"]*g_l_c_gan1+\
                lambda_dict["lambda1"]*g_l_data2+lambda_dict["lambda1"]*g_l_data3
            g_loss.backward()
            optimizer_g.step()
            # print(g_loss)

            epoch_g_loss+=g_loss.item()    #生成器总损失
            epoch_single_g_loss+=g_l_c_gan1.item()  #gan损失
            epoch_tf_loss += g_l_data1.item()  # gt重构损失

        t_epoch_finish=time.time()
        Epoch_D_Loss=epoch_d_loss/(lambda_dict["lambda2"]*2*data_len)
        Epoch_G_Loss=epoch_g_loss/data_len
        Epoch_Single_G_Loss=epoch_single_g_loss/data_len
        # Epoch_nd_Loss = epoch_nd_loss / data_len
        Epoch_tf_Loss=epoch_tf_loss/data_len

        print("------------")
        print("epoch {}  || Epoch_D_Loss:{:.4f}  || Epoch_G_Loss:{:.4f} ||  Epoch_Single_G_Loss:{:.4f}  ||  Epoch_tf_Loss:{:.4f}".format(
            epoch,
            epoch_d_loss/(lambda_dict["lambda2"]*2*data_len),
            epoch_g_loss/data_len,
            epoch_single_g_loss/data_len,
            epoch_tf_loss / data_len,
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
        # g_losses.append(Epoch_D_Loss)
        g_losses.append(epoch_d_loss/(lambda_dict["lambda2"]*2*data_len))
        d_losses.append(Epoch_G_Loss)
        single_gan_losses.append(Epoch_Single_G_Loss)
        general_losses.append(Epoch_tf_Loss)

        # 输出损失日志
        plot_log(
          {
              "G":g_losses,
              "D":d_losses,
              "SG":single_gan_losses,
              "GENERAL":general_losses
          },
          save_model_name+str(epoch)
       )

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