import math

import torch
import torch.nn as nn


class DN_Net(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(DN_Net, self).__init__()
        #对噪声图像编码
        self.noise_encoder = Encoder(input_channels)
        #对平滑过后的噪声分布1进行编码
        self.noiseone_encoder = NoiseOneEncoder(input_channels)
        #对平滑过后的噪声分布2进行编码
        self.noisetwo_encoder = NoiseTwoEncoder(input_channels)

        self.clean_decoder = CleanDecoder(input_channels)
        self.noise1_decoder = NoiseOneDecoder(input_channels)
        #噪声移除解码器
        self.noise2_decoder=NoiseTwoDecoder(input_channels)

        #定义一个实例变量placeholder并将其初始化为None，可以再类方法里面使用
        self.placeholder = None

    def forward(self, noise_img,noise1,noise2):
        #分别使用噪声图像编码器，和有雾图像编码器进行编码
        noise_features = self.noise_encoder(noise_img)
        noise1_features = self.noiseone_encoder(noise1)
        noise2_features = self.noisetwo_encoder(noise2)
        #对噪声分布解码
        noise1=self.noise1_decoder(noise1_features)
        noise2 = self.noise1_decoder(noise2_features)
        gt_recon=self.clean_decoder(noise_features, noise1_features)

        return noise1, noise2, gt_recon

    def test(self,noise_img,noise1,noise2):
        # 分别使用噪声图像编码器，和有雾图像编码器进行编码
        noise_features = self.noise_encoder(noise_img)
        noise1_features = self.noiseone_encoder(noise1)
        noise2_features = self.noisetwo_encoder(noise2)
        # 对噪声分布解码
        noise1 = self.noise1_decoder(noise1_features)
        noise2 = self.noise1_decoder(noise2_features)
        gt_recon = self.clean_decoder(noise_features, noise1_features)

        return noise1, noise2, gt_recon

    def test1(self,noise_img, gt_img):
        noise_features = self.noise_encoder(noise_img)
        gt_features = self.gt_encoder(gt_img)
        reconstruct_gt=self.noise_move_decoder(noise_features, gt_features)   #阴影移除联合解码器Js2sf


        return reconstruct_gt

    def test_pair(self,noise_img,gt_img):
        noise_features = self.noise_encoder(noise_img)
        gt_features = self.gt_encoder(gt_img)

        if self.placeholder is None or self.placeholder.size(0)!=noise_features.size(0):
            self.placeholder={}
            for key in noise_features.keys():
                self.placeholder[key]=torch.zeros(noise_features.shape,requires_grad=False).to(
                    torch.device(noise_features.device)
                )

        rec_noise=self.noise_encoder(noise_features,self.placeholder)
        rec_clean=self.clean_decoder(self.placeholder,gt_features)

        return rec_clean,rec_noise


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

class Encoder(nn.Module):
    def __init__(self,input_channels=3):
        super(Encoder, self).__init__()
        self.conv1=Cvi(input_channels,64)
        self.conv2 = Cvi(64, 128, before="LReLU", after="BN")
        self.conv3 = Cvi(128, 256, before="LReLU", after="BN")
        self.conv4 = Cvi(256, 512, before="LReLU")

    def forward(self, x):
        x1=self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4=self.conv4(x3)

        feature_dic = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,

        }

        return feature_dic


class NoiseOneEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(NoiseOneEncoder, self).__init__()
        self.conv1 = Cvi(input_channels, 64)
        self.conv2 = Cvi(64, 128, before="LReLU", after="BN")
        self.conv3 = Cvi(128, 256, before="LReLU", after="BN")
        self.conv4 = Cvi(256, 512, before="LReLU")

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        feature_dic = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,

        }

        return feature_dic

class NoiseTwoEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(NoiseTwoEncoder, self).__init__()
        self.conv1 = Cvi(input_channels, 64)
        self.conv2 = Cvi(64, 128, before="LReLU", after="BN")
        self.conv3 = Cvi(128, 256, before="LReLU", after="BN")
        self.conv4 = Cvi(256, 512, before="LReLU")

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        feature_dic = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,

        }

        return feature_dic


class CleanDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(CleanDecoder, self).__init__()
        self.conv1 = CvTi(512, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(512, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(256, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(128, output_channels, before="ReLU", after="Tanh")


    def forward(self, noise, dn1):

        x4 = (noise["x4"]-dn1["x4"])/2.0
        x3 = self.conv1(x4)
        cat3 = torch.cat([x3,(noise["x3"]-dn1["x3"])/2.0], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2,(noise["x2"]-dn1["x2"])/2.0], dim=1)
        x1= self.conv3(cat2)
        cat1 = torch.cat([x1,(noise["x1"]-dn1["x1"])/2.0], dim=1)
        x = self.conv4(cat1)

        return x
class NoiseOneDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(NoiseOneDecoder, self).__init__()
        self.conv1 = CvTi(512, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(512, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(256, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(128, output_channels, before="ReLU", after="Tanh")


    def forward(self, noise1):

        x4 = noise1["x4"]
        x3 = self.conv1(x4)
        cat3 = torch.cat([x3,noise1["x3"]], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2,noise1["x2"]], dim=1)
        x1= self.conv3(cat2)
        cat1 = torch.cat([x1,noise1["x1"]], dim=1)
        x = self.conv4(cat1)

        return x
class NoiseTwoDecoder(nn.Module):
    def __init__(self,output_channels=3):
        super(NoiseTwoDecoder, self).__init__()
        self.conv1 = CvTi(1024, 256, before="ReLU", after="BN")
        self.conv2 = CvTi(512, 128, before="ReLU", after="BN")
        self.conv3 = CvTi(256, 64, before="ReLU", after="BN")
        self.conv4 = CvTi(64, output_channels, before="ReLU", after="Tanh")


    def forward(self, noise2):

        x4 = noise2["x4"]
        x3 = self.conv1(x4)
        cat3 = torch.cat([x3,noise2["x3"]], dim=1)
        x2 = self.conv2(cat3)
        cat2 = torch.cat([x2,noise2["x2"]], dim=1)
        x1= self.conv3(cat2)
        cat1 = torch.cat([x1,noise2["x1"]], dim=1)
        x = self.conv4(cat1)

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
            x=self.before(x)
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
    # l2=nn.L2Loss()
    model = DN_Net()
    noise_img = torch.randn(1, 3, 256, 256)
    noise1 = torch.randn(1, 3, 256, 256)
    noise2 = torch.randn(1, 3, 256, 256)
    gt_img = torch.randn(1, 3, 256, 256)
    noise1, noise2, gt_recon= model(noise_img, noise1,noise2)
    print(noise1.shape, noise2.shape, gt_recon.shape)

    # size(3,3,256,256)
    input=torch.ones(size)


