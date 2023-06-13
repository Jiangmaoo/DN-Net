import math

import torch
import torch.nn as nn
import numpy as np
#
# 在这个解码器模型中，我们使用四个反卷积层对特征向量进行解码，输出图像。
# 每个反卷积层后面都有一个批归一化层和一个ReLU激活函数。最后一个反卷积层的输出通道数等于输出图像的通道数。
#
# 现在，我们可以定义整个网络模型。我们需要实例化两个编码器，两个解码器以及两个掩码，
# 其中一个掩码对应相似像素，另一个对应不相似像素。然后，我们将输入图像馈送到两个编码器中，得到相似和不相似特征向量。
# 接下来，我们将这些特征向量传递给相应的解码器，得到相似和不相似图像。
# 最后，我们计算相似图像与相应掩码之间的L1损失，以及不相似图像与gt之间的L1损失，并将它们加起来作为总损失。
#


class Net(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):   #为什么是这个数字
        super(Net, self).__init__()
        #本体特征编码器
        self.encoder_similar = Encoder(input_channels, output_channels)

        #噪声特征编码器
        self.encoder_different = Encoder(input_channels, output_channels)
        #重建图像解码器
        self.decoder_similar = Decoder(output_channels, input_channels)
        #噪声解码器
        self.decoder_different = Decoder(output_channels, input_channels)

        self.mask_similar = nn.Parameter(torch.ones(1, output_channels, 1, 1), requires_grad=True)
        self.mask_different = nn.Parameter(torch.ones(1, output_channels, 1, 1), requires_grad=True)

    def forward(self, x, gt):
        # Encode input image into similar and different feature vectors
        feature_similar = self.encoder_similar(x)
        feature_different = self.encoder_different(x)

        # Decode similar and different feature vectors into images
        similar = self.decoder_similar(feature_similar * self.mask_similar)
        different = self.decoder_different(feature_different * self.mask_different)

        # Calculate L1 loss between similar image and similar mask
        loss_similar = nn.L1Loss()(similar * self.mask_similar, x * self.mask_similar)

        # Calculate L1 loss between different image and ground truth
        loss_different = nn.L1Loss()(different * self.mask_different, gt)

        # Calculate total loss
        loss = loss_similar + loss_different

        return similar, different, loss

    #模型测试
    def test(self,input_img):
        feature_dic1=self.domain1_encoder(input_img)    #阴影编码器
        general_dic1=self.general_encoder(input_img)    #共同特征编码器
        reconstruct_input=self.joint_decoderT(feature_dic1,general_dic1)    #阴影移除联合解码器Js2sf

        return reconstruct_input
    def test_pair(self,input_img):
        feature_dic1=self.domain1_encoder(input_img)
        general_dic1=self.general_encoder(input_img)

        if self.placeholder is None or self.placeholder["x1"].size(0)!=feature_dic1["x1"].size(0):
            self.placeholder={}
            for key in feature_dic1.keys():
                self.placeholder[key]=torch.zeros(feature_dic1[key].shape,requires_grad=False).to(
                    torch.device(feature_dic1["x1"].device)
                )
        rec_by_tg1=self.joint_decoder(self.placeholder,general_dic1)

        rec_by_td1=self.joint_decoder(feature_dic1,self.placeholder)

        reconstruct_tf=self.joint_decoderT(feature_dic1,general_dic1)

        return reconstruct_tf,rec_by_tg1,rec_by_td1



class Encoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        return x

#初始化权重
def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname=m.__class__.__name__
        if (classname.find("Conv")==0 or classname.find("Linear")==0 or hasattr(m,"weight")):
            if init_type=="gaussian":
                nn.init.normal_(m.weight,0.0,0.02)
            elif init_type=="xavier":
                nn.init.xavier_normal_(m.weight,gain=math.sqrt(2))
            elif init_type=="kaiming":
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
            elif init_type=="orthogonal":
                nn.init.orthogonal_(m.weight,gain=math.sqrt(2))
            elif init_type=="default":
                pass
            else:
                assert 0,"Unsupported initialization:{}".format(init_type)
            if hasattr(m,"bias") and m.bias is not None:
                nn.init.constant_(m.bias,0.0)
    return init_fun
#卷积
class Cvi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(Cvi,self).__init__()

        #初始化卷积
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)

        #初始化卷积参数
        self.conv.apply(weights_init("gaussian"))

        #卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)   #归一化
        elif after=="Tanh":
            self.after=torch.tanh #tanh激活函数（-1到1S型）
        elif after=="sigmoid":
            self.after=torch.sigmoid    #sigmoid激活函数（0到1S型）

        #卷积前进行的操作
        if before=="ReLU":
            self.after=nn.ReLU(inplace=True)  #ReLU激活函数（<0时=0；>0时等于自身)(inplace=True,节省反复申请与释放内存的空间和时间)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=False)  #LeakyReLu激活函数（<0时斜率为0.2）

    def forward(self,x):
        if hasattr(self,"before"):
            x=self.before(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x
#逆卷积
class CvTi(nn.Module):
    def __init__(self,in_channels,out_channels,before=None,after=False,kernel_size=4,stride=2,
                 padding=1,dilation=1,groups=1,bias=False):
        super(CvTi, self).__init__()

        #初始化逆卷积
        self.conv=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                     stride=stride,padding=padding)
        #初始化逆卷积权重
        self.conv.apply(weights_init("gaussian"))

        # 卷积后进行的操作
        if after=="BN":
            self.after=nn.BatchNorm2d(out_channels)
        elif after=="Tanh":
            self.after=torch.tanh
        elif after=="sigmoid":
            self.after=torch.sigmoid

        #卷积前进行的操作
        if before=="ReLU":
            self.before=nn.ReLU(inplace=True)
        elif before=="LReLU":
            self.before=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        if hasattr(self,"before"):
            x=self.conv(x)
        x=self.conv(x)
        if hasattr(self,"after"):
            x=self.after(x)
        return x
#鉴别器
class Discriminator(nn.Module):
    def __init__(self,input_channels=4):
        super(Discriminator, self).__init__()
        self.cv0=Cvi(input_channels,64)
        self.cv1=Cvi(64,128,before="LReLU",after="BN")
        self.cv2 = Cvi(128, 256, before="LReLU", after="BN")
        self.cv3 = Cvi(256, 512, before="LReLU", after="BN")
        self.cv4 = Cvi(512, 1, before="LReLU", after="sigmoid")

    def forward(self,x):
        x0=self.cv0(x)
        x1=self.cv1(x0)
        x2=self.cv2(x1)
        x3=self.cv3(x2)
        out=self.cv4(x3)

        return out


if __name__=='__main__':
    #BCHW
    size=(3,3,256,256)
    input1=torch.ones(size)
    input2=torch.ones(size)
    l2=nn.L2Loss()

    size(3,3,256,256)
    input=torch.ones(size)

