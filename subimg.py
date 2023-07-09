#邻居子采样图像模块。原尺寸

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from models.MeanFilter import MeanFilter
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

operation_seed_counter=0
device="cuda:0" if torch.cuda.is_available() else "cpu"

#返回一个CUDA设备上的生成器
def get_generator():
    global operation_seed_counter  #定义一个全局变量，用于生成操作的种子
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cpu")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

#用于将输入张量x进行空间到深度的转换，再一定程度上增加网络的非线性容量，处进行信息的流动和特征的重用
def space_to_depth(x, block_size):
    n, c, h, w = x.size()  #获取张量x的维度信息
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=1,padding=1)  #使用unfold函数对输入张量进行展开操作，将图像歌城block*block的块
    return unfolded_x.view(n, c * block_size**2, h, w)   #实现空间到深度的转换。3 1 1

#为给定图像生成一对掩码
def generate_mask_pair(img):
    # prepare masks (N x C x H x W)
    n, c, h, w = img.shape
    print("img.shape:{}".format(img.shape))
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
                  size=(n * h * w , ),
                  generator=get_generator(),
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

def calNoise(images):
    mask1, mask2 = generate_mask_pair(images) #mask1.shape:torch.Size([65536]) 65536=256*256
    noisy_sub1 = generate_subimages(images, mask1)  #noisy_sub1.shape:torch.Size([1, 3, 128, 128])
    noisy_sub2 = generate_subimages(images, mask2)

    # print("noisy_sub1.shape:{}".format(noisy_sub1.shape))

    mean_filter = MeanFilter(kernel_size=5)
    denoise1 = mean_filter(noisy_sub1) #torch.Size([1, 3, 128, 128])
    denoise2 = mean_filter(noisy_sub2)

    # print(denoise1.shape)
    # print("image.shape:{}".format(images.shape))#image.shape:torch.Size([1, 3, 256, 256])

    # 计算噪声图像和干净图像的差异
    diff1 = torch.abs(images - denoise1)
    # 根据差异计算掩码
    threshold = 0.05  # 可以根据具体情况调整阈值
    noise1 = (diff1 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0
    diff2 = torch.abs(images - denoise2)
    noise2 = (diff2 > threshold).float()  # 大于阈值的位置被置为 1，否则为 0

    fig = plt.figure(figsize=(6, 9))
    axes = fig.subplots(3, 2)

    axes[0, 0].imshow(noisy_sub1.squeeze().permute(1, 2, 0))
    axes[0, 1].imshow(noisy_sub2.squeeze().permute(1, 2, 0))

    axes[1, 0].imshow(denoise1.squeeze().permute(1, 2, 0))
    axes[1, 1].imshow(denoise2.squeeze().permute(1, 2, 0))

    axes[2, 0].imshow(noise1.squeeze().permute(1, 2, 0))
    axes[2, 1].imshow(noise2.squeeze().permute(1, 2, 0))

    # 关闭坐标轴
    for ax in axes.flatten():
        ax.axis('off')

    # 调整子图间距
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # 显示图像
    plt.show()

    return noise1,noise2

if __name__=="__main__":
    # input_data = torch.randn(1, 3, 256, 256)  # Random input tensor with shape (batch_size, channels, height, width)
    # noise1, noise2 = calNoise(input_data)
    # print(noise1.shape)
    # print(noise2.shape)
    # img = Image.open('dataset/0004.png').convert('RGB')

    #读取图像
    # img=cv2.imread('dataset/0036.png')
    # 图像预处理和转换
    transform = transforms.Compose([
        transforms.Resize((383, 383)),  # 调整图像尺寸
        transforms.ToTensor()  # 转换为张量
    ])

    # 读取图像
    image_path = 'dataset/0312.png'
    image = Image.open(image_path)

    # 进行预处理和转换，并添加批次维度
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)  # 在第0维度添加批次大小

    # 显示张量的形状
    print(tensor_image.shape)


    noise1, noise2 = calNoise(tensor_image)
    print(noise1.shape)