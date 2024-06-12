import sys
sys.path.append('..')

import math
import torch
from torch import nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from nn.TrjSR_utils import traj2cell_test_lr, draw_lr

class TrjSR(nn.Module):
    def __init__(self, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr, traj_embedding_emb):
        super(TrjSR, self).__init__()
        self.g = MyGenerator()
        self.d = MyDiscriminator(traj_embedding_emb)
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.imgsize_x_lr = imgsize_x_lr
        self.imgsize_y_lr = imgsize_y_lr
        self.pixelrange_lr = pixelrange_lr

    def forward(self, trajs_img):
        embs = self.d(self.g(trajs_img))
        return embs
    
    @torch.no_grad()
    def interpret(self, trajs_img):
        device = next(self.parameters()).device
        embs = self.forward(trajs_img.to(device))
        return embs
       
    @torch.no_grad()
    def trajsimi_interpret(self, trajs_img, trajs_img2):
        device = next(self.parameters()).device
        embs = self.forward(trajs_img.to(device))
        embs2 = self.forward(trajs_img2.to(device))
        dists = F.pairwise_distance(embs, embs2, p = 1)
        return dists.detach().cpu().tolist()


def input_processing(trajs, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    # src : 3D list
    trajs_img = []
    for traj in trajs:
        test_traj = traj2cell_test_lr(traj, lon_range, lat_range, 
                                    imgsize_x_lr, imgsize_y_lr, pixelrange_lr)
        traj_img = ToTensor()(draw_lr(test_traj, imgsize_x_lr, imgsize_y_lr, pixelrange_lr))

        trajs_img.append(traj_img)
    trajs_img = torch.stack(trajs_img)
    return trajs_img


# for trajsimi 
def collate_fn(batch, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    src, src2 = zip(*batch)
    trajs_img = input_processing(src, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr)
    trajs_img2 = input_processing(src2, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr)
    return trajs_img, trajs_img2
    
# for knn
def collate_fn_single(src, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    trajs_img = input_processing(src, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr)
    return trajs_img


class MyGenerator(nn.Module):
    def __init__(self, scale_factor = 2):
        kernels = 16
        upsample_block_num = int(math.log(scale_factor, 2))

        super(MyGenerator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, kernels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(kernels)
        self.block3 = ResidualBlock(kernels)
        self.block4 = ResidualBlock(kernels)
        self.block5 = ResidualBlock(kernels)
        self.block6 = ResidualBlock(kernels)
        self.block7 = nn.Sequential(
            nn.Conv2d(kernels, kernels, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernels)
        )
        block8 = [UpsampleBLock(kernels, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(kernels, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        # block5 = self.block5(block4)
        # block6 = self.block6(block5)
        block7 = self.block7(block4)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class MyDiscriminator(nn.Module):
    def __init__(self, traj_embedding_emb):
        super(MyDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1, 16, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            )
        self.out_layer = nn.Linear(32*40, traj_embedding_emb)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 32*40)
        x = self.out_layer(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # identity_data = x
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        # output = torch.add(output, identity_data)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class lowdim(nn.Module):
    def __init__(self):
        super(lowdim, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
            # nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 32*41)
        # x = self.fc1(x)
        # x2 = self.fc2(x1)
        # x = self.fc3(x)
        return x
