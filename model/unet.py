import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.autograd.functional import jacobian
import gc

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)
        self.down2 = Down(2 * in_channels, 4 * in_channels)
        self.down3 = Down(4 * in_channels, 8 * in_channels)
        self.up1   = Up(8 * in_channels, 4 * in_channels)
        self.up2   = Up(4 * in_channels, 2 * in_channels)
        self.up3   = Up(2 * in_channels, in_channels)
        self.final = nn.Conv3d(in_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x, x2)
        x  = self.up3(x, x1)
        x  = self.final(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

# 约15s执行到池化
# 手工实现MaxPool3dMS代替Python-MaxPool3d
class MaxPool3dMPS(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool3dMPS, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        #print("MaxPool3dMPS启动¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥")
        # 确保输入x是5D张量
        if x.dim() != 5:
            raise ValueError("Expected 5D tensor as input")

        # 计算输出的形状
        d = (x.size(2) - self.kernel_size + 2 * self.padding) // self.stride + 1
        h = (x.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
        w = (x.size(4) - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 初始化输出张量
        output = torch.zeros((x.size(0), x.size(1), d, h, w), device=x.device)

        # 执行最大池化
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    region = x[:, :, i*self.stride:i*self.stride+self.kernel_size,
                                    j*self.stride:j*self.stride+self.kernel_size,
                                    k*self.stride:k*self.stride+self.kernel_size]
                    # 使用.reshape()来确保操作后的张量内存连续
                    max_values = region.reshape(x.size(0), x.size(1), -1).max(dim=2)[0]
                    output[:, :, i, j, k] = max_values

        del x
        gc.collect()

        return output

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            #nn.MaxPool3d(2),
            #手工实现
            MaxPool3dMPS(2),
            ConvBlock(in_channels, out_channels)
        )

'''# 约23s执行到此步
#手工
class ConvTranspose3dMPS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, bias=False):
        super(ConvTranspose3dMPS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 初始化权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        print("ConvTranspose3dMPS启动¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥")
        # 手动进行三维上采样
        B, C, D, H, W = x.size()
        x = x.view(B, C, D, 1, H, 1, W, 1)
        x = x.repeat(1, 1, 1, self.stride, 1, self.stride, 1, self.stride)
        x = x.view(B, C, D * self.stride, H * self.stride, W * self.stride)

        # 手动进行卷积操作
        output = torch.zeros(
            (B, self.out_channels, D * self.stride, H * self.stride, W * self.stride),
            device=x.device
        )
        # 对每个输出通道和输入通道对进行卷积
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                # 对输入张量进行卷积核大小的切片
                for d in range(D * self.stride):
                    for h in range(H * self.stride):
                        for w in range(W * self.stride):
                            # 计算卷积核覆盖的区域
                            start_d = d - self.padding
                            end_d = start_d + self.kernel_size
                            start_h = h - self.padding
                            end_h = start_h + self.kernel_size
                            start_w = w - self.padding
                            end_w = start_w + self.kernel_size
                            # 确保索引在合法范围内
                            if start_d >= 0 and end_d <= D and start_h >= 0 and end_h <= H and start_w >= 0 and end_w <= W:
                                # 执行卷积操作
                                output[:, i, d, h, w] += torch.sum(
                                    x[:, j, start_d:end_d, start_h:end_h, start_w:end_w] * self.weight[i, j],
                                    dim=(1, 2, 3)
                                )
        # 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output'''

# 算法优化后的手工复现
class ConvTranspose3dMPS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, bias=False):
        super(ConvTranspose3dMPS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 初始化权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        print("ConvTranspose3dMPS启动¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥")
        # 手动进行三维上采样
        B, C, D, H, W = x.size()
        x = x.view(B, C, D, 1, H, 1, W, 1)
        x = x.repeat(1, 1, 1, self.stride, 1, self.stride, 1, self.stride)
        x = x.view(B, C, D * self.stride, H * self.stride, W * self.stride)

        # 使用FFT进行卷积操作
        output = torch.zeros(
            (B, self.out_channels, D * self.stride, H * self.stride, W * self.stride),
            device=x.device
        )

        # 对每个输出通道和输入通道对进行卷积
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                # 执行FFT卷积操作
                fft_x = torch.fft.fftn(x[:, j], dim=(1, 2, 3))
                fft_weight = torch.fft.fftn(self.weight[i, j], s=x.shape[-3:])
                fft_output = fft_x * fft_weight
                output[:, i] += torch.fft.ifftn(fft_output, dim=(1, 2, 3)).real
                
        del fft_x, fft_weight, fft_output
        gc.collect()

        # 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        
        del x
        gc.collect()
        
        return output

'''class ConvTranspose3dMPS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, bias=False):
        super(ConvTranspose3dMPS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 初始化权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        #self.shapes =

    #@staticmethod
    #def forward(ctx, x, weight, shapes):
    #def forward(self, x, weight, shapes):
    def forward(self, x):
        # 在这里直接使用 self.weight 和 self.shapes
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        B,in_C,in_D,in_H,in_W = x.shape
        out_C,_,k_D,k_H,k_W = self.weight.shape
        p_D,p_H,p_W = self.shapes[0].tolist()#padding
        s_D,s_H,s_W = self.shapes[1].tolist()#stride
        d_D,d_H,d_W = self.shapes[2].tolist()#dilation
        out_D,out_H,out_W = self.shapes[3].tolist()#shape_out
        groups,_,_ = self.shapes[4].tolist()
        self.shapes = [torch.tensor([p_D,p_H,p_W]), torch.tensor([s_D,s_H,s_W]), torch.tensor([d_D,d_H,d_W]), torch.tensor([out_D,out_H,out_W]), torch.tensor([groups])]
        # 使用 self.shapes
        weight2d = self.weight.view(out_C,-1,k_H,k_W)
        unfold_weight = torch.eye(k_D,k_D).to(device).view(k_D,1,k_D,1)
        x2d = F.conv2d(x.view(-1,1,in_D,in_H*in_W),unfold_weight,padding=(p_D,0),stride=(s_D,1),dilation=(d_D,1))
        x2d_ = x2d.view(B,in_C,k_D,out_D,in_H,in_W).permute(0,3,1,2,4,5).reshape(B*out_D,in_C*k_D,in_H,in_W)
        out = F.conv2d(x2d_,weight2d,padding=(p_H,p_W),stride=(s_H,s_W),dilation=(d_H,d_W),groups=groups).view(B,out_D,out_C,out_H,out_W).permute(0,2,1,3,4)
        self.save_for_backward(x2d_,weight2d,unfold_weight,self.shapes)
        return out
    #@staticmethod
    #def backward(ctx, gradient):
    def backward(self, gradient):
        #x2d_,weight2d,unfold_weight,shapes = ctx.saved_tensors
        x2d_,weight2d,unfold_weight,shapes = self.saved_tensors
        B,in_C,in_D,in_H,in_W = x2d_.shape
        out_C,_,k_D,k_H,k_W = weight2d.shape
        p_D,p_H,p_W = shapes[0].tolist()#padding
        s_D,s_H,s_W = shapes[1].tolist()#stride
        d_D,d_H,d_W = shapes[2].tolist()#dilation
        out_D,out_H,out_W = shapes[3].tolist()#shape_out
        groups,_,_ = shapes[4].tolist()

        outback = gradient.permute(0,2,1,3,4).reshape(B*out_D,out_C,out_H,out_W)
        x2d_grad_ = jacobian(lambda x: (F.conv2d(x,weight2d,padding=(p_H,p_W),dilation=(d_H,d_W),stride=(s_H,s_W),groups=groups)-outback)\
                            .pow(2).mul(0.5).sum(),torch.zeros(B*out_D,in_C*k_D,in_H,in_W))
        x2d_grad = x2d_grad_.reshape(B,out_D,in_C,k_D,in_H,in_W).permute(0,2,3,1,4,5).reshape(B*in_C,k_D,out_D,in_H*in_W)
        x_grad_ = jacobian(lambda x: (F.conv2d(x,unfold_weight,padding=(p_D,0),dilation=(d_D,1),stride=(s_D,1))-x2d_grad)\
                            .pow(2).mul(0.5).sum(),torch.zeros(B*in_C,1,in_D,in_H*in_W))
        x_grad = x_grad_.view(B,in_C,in_D,in_H,in_W)
        w_grad = jacobian(lambda w: (F.conv2d(x2d_,w,padding=(p_H,p_W),dilation=(d_H,d_W),stride=(s_H,s_W),groups=groups)-outback).pow(2).mul(0.5).sum(), torch.zeros(out_C,in_C*k_D//groups,k_H,k_W)).view(out_C,in_C//groups,k_D,k_H,k_W)

        return x_grad,w_grad,None #shapes has no grad'''

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            #nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            ConvTranspose3dMPS(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x
