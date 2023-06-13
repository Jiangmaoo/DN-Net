# 假设原始干净图像和加了雾的图像已经加载为名为 gt 和 noise_img 的 Tensor 对象

# 计算噪声图像和干净图像的差异
diff = torch.abs(noise_img - gt)

# 根据差异计算掩码
threshold = 0.05 # 可以根据具体情况调整阈值
mask = (diff > threshold).float() # 大于阈值的位置被置为 1，否则为 0

# 将 mask 作为参数传入模型进行训练或测试
