import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFilter(nn.Module):
    def __init__(self, kernel_size):
        super(MeanFilter, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size) / (self.kernel_size ** 2)
        kernel = kernel.to(x.device)

        output = torch.zeros_like(x)
        for c in range(channels):
            output[:, c, :, :] = F.conv2d(x[:, c, :, :].unsqueeze(1), kernel, padding=self.kernel_size // 2).squeeze(1)

        return output

if __name__=="__main__":
    # Example usage
    input_data = torch.randn(1, 3, 256, 256)  # Random input tensor with shape (batch_size, channels, height, width)
    mean_filter = MeanFilter(kernel_size=5)  # Create an instance of the Mean filter
    output = mean_filter(input_data)  # Apply the Mean filter to the input tensor

    print(output.shape)  # Output shape will be (1, 1, 32, 32)
