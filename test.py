import argparse
import math

import cv2
import numpy as np
import torch
import os

import torch.nn.functional as F

from collections import OrderedDict

from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models.DN_network import DN_Net
from train import calNoise
from utils.data_loader import ImageDataset, ImageTransform, ImageTransformOwn, make_data_path_list
from PIL import Image

torch.manual_seed(44)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parser():
    parser = argparse.ArgumentParser(
        prog='D-N Net',
        usage='python3 main.py',
        description='This module demonstrates NOISE removal using D-N Net.',
        add_help=True
    )

    parser.add_argument('-l', '--load', type=str, default="D-N-Net_G1_240", help='the number of checkpoints')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='file path of image you want to test')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=512)   #这样输入网络大小就是512
    parser.add_argument('-rs', '--resized_size', type=int, default=256)

    return parser


def fix_model_state_dict(state_dict):
    # remove 'module.' of data parallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def check_dir():
    if not os.path.exists('./test_result'):
        os.mkdir('./test_result')
    # if not os.path.exists('./test_result/NOISE_removal_image'):
    #     os.mkdir('./test_result/NOISE_removal_image')
    # if not os.path.exists('./test_result/grid'):
    #     os.mkdir('./test_result/shadow_removal_grid_images')


def un_normalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def test_own_image(g1, g2, path, out_path, size, img_transform):
    img = Image.open(path).convert("RGB")
    width, height = img.width, img.height
    img = img.resize((size, size), Image.LANCZOS)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    g1.to(device)
    g2.to(device)

    # use GPU in parallel
    if device == 'cuda':
        g1 = torch.nn.DataParallel(g1)
        g2 = torch.nn.DataParallel(g2)
        print("parallel mode")

    print("device:{}".format(device))

    g1.eval()
    g2.eval()

    with torch.no_grad():
        detected_shadow = g1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device("cpu"))
        concat = torch.cat([img, detected_shadow], dim=1)
        noise_removal_image = g2(concat.to(device))
        noise_removal_image = noise_removal_image.to(torch.device("cpu"))

        grid = make_grid(torch.cat(
            [
                un_normalize(img),
                un_normalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1)),
                un_normalize(noise_removal_image)
            ],
            dim=0
        ))

        save_image(grid, out_path + "/grid_" + path.split("/")[-1])

        detected_shadow = transforms.ToPILImage(mode="L")(un_normalize(detected_shadow)[0, :, :, :])
        detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)
        detected_shadow.save(out_path + "/detected_shadow_" + path.split("/")[-1])

        noise_removal_image = transforms.ToPILImage(mode="RGB")(un_normalize(noise_removal_image)[0, :, :, :])
        noise_removal_image = noise_removal_image.resize((width, height), Image.LANCZOS)
        noise_removal_image.save(out_path + '/NOISE_removal_image' + path.split('/')[-1])


def test_dataset_image(g1, test_dataset):
    check_dir()#检查并创建保存结构的文件夹
    device = "cuda" if torch.cuda.is_available() else "cpu"  #根据是否有GPU，选择使用GPU或者CPU作为设备
    g1.to(device)   #把模型g1移动到选择的设备上
    g1.eval()   #将其设置为评估模式 （即不进行梯度更新）

    #len=test_dataset.__len__()
    PSNRs = []
    SSIMs = []

    for n, (img,gt) in enumerate([test_dataset[i] for i in range(test_dataset.__len__())]):
        print(test_dataset.img_list["path_A"][n])

        img = torch.unsqueeze(img, dim=0)
        noise1, noise2 = calNoise(img, "cpu")
        gt = torch.unsqueeze(gt, dim=0)

        with torch.no_grad():   #上下文管理器，避免计算梯度，加快图像处理速度
            reconstruct_tf = g1.test1(img.to(device),noise1.to(device),noise2.to(device))
            reconstruct_tf = reconstruct_tf.to(torch.device("cpu")) #使用g1神经网络对图像进行阴影处理，并将结果转换为CPU张量
        # 计算并打印PSNR和SSIM指标
        psnr_val = psnr(gt, reconstruct_tf)
        ssim_val = ssim(gt, reconstruct_tf)
        # print("PSNR: {:.4f}, SSIM: {:.4f}".format(psnr_val, ssim_val))

        #将去除阴影的图像转换为PIL图像并保存到“./test_result”文件夹中，其中文件名与输入图像的文件名相同
        noise_removal_image = transforms.ToPILImage(mode="RGB")(un_normalize(reconstruct_tf)[0, :, :, :])
        noise_removal_image.save("./test_result" + "/" + test_dataset.img_list["path_A"][n].split("/")[-1])

        #将真实图像转换为PIL图像并保存到“./test_gt”文件夹中，其中文件名与输入图像文件名相同
        #gt = transforms.ToPILImage(mode="RGB")(un_normalize(gt)[0, :, :, :])
        #gt.save("./test_gt" + "/" + test_dataset.img_list["path_C"][n].split("/")[-1])

        PSNRs.append(psnr_val)
        SSIMs.append(ssim_val)

    PSNR=np.mean(PSNRs)
    SSIM=np.mean(SSIMs)

    # 将以下代码添加到保存图像的代码之前
    # 保存PSNR和SSIM指标到文件中
    with open("./test_result/psnr_ssim.txt", "a") as f:
        f.write(
            "平均值：{}\tPSNR: {:.4f}, SSIM: {:.4f}\n".format(test_dataset.img_list["path_A"][n].split("/")[-1], PSNR,
                                                      SSIM))

def test(parser):
    # g1 = DN_Net(input_channels=3, output_channels=1)
    g1 = DN_Net(input_channels=3, output_channels=3)

    # 是否加载模型
    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        g1_weights = torch.load('./checkpoints/' + str(parser.load) + '.pth')
        g1.load_state_dict(fix_model_state_dict(g1_weights))

    mean = (0.5,)
    std = (0.5,)

    size = parser.image_size
    crop_size = parser.crop_size
    resized_size = parser.resized_size

    # 测试非数据集中的图片
    if parser.image_path is not None:
        print('test ' + parser.image_path)
        test_own_image(g1, parser.image_path, parser.out_path, resized_size,
                       img_transform=ImageTransformOwn(size=size, mean=mean, std=std))
    # 测试ISTD数据集中的图片
    else:
        test_img_list = make_data_path_list(phase='test')
        test_dataset = ImageDataset(img_list=test_img_list,
                                    img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                    phase='test_no_crop')
        test_dataset_image(g1, test_dataset)

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)



if __name__ == "__main__":
    main_parser = get_parser().parse_args()
    test(main_parser)
