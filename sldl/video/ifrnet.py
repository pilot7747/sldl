import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        1,
    )
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


def resize(x, scale_factor: float):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.PReLU(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(
            out[:, -self.side_channels:, :, :]
        )
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(
            out[:, -self.side_channels:, :, :]
        )
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1), convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1), convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1), convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1), convrelu(96, 96, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192 + 1, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True),
        )

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(220, 216),
            ResBlock(216, 32),
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True),
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(148, 144),
            ResBlock(144, 32),
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True),
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(100, 96),
            ResBlock(96, 32),
            nn.ConvTranspose2d(96, 8, 4, 2, 1, bias=True),
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class IFRNet(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(IFRNet, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()

    def forward(self, img0, img1, embt, scale_factor: float = 1.0):
        mean_ = (
            torch.cat([img0, img1], 2)
            .mean(1, keepdim=True)
            .mean(2, keepdim=True)
            .mean(3, keepdim=True)
        )
        img0 = img0 - mean_
        img1 = img1 - mean_

        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (
            1.0 / scale_factor
        )
        up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (
            1.0 / scale_factor
        )
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0 / scale_factor))
        up_res_1 = resize(up_res_1, scale_factor=(1.0 / scale_factor))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred


class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""

    def __init__(self, dims, divisor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


@torch.no_grad()
def ifnet_inference(model, image1, image2, device):
    img0_np = np.asarray(image1)
    img1_np = np.asarray(image2)

    img0 = (
        (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0)
        .unsqueeze(0)
        .to(device)
    )
    img1 = (
        (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0)
        .unsqueeze(0)
        .to(device)
    )

    need_pad = False

    for dim_size in img0_np.shape:
        if dim_size % 16 != 0:
            need_pad = True
            break

    if need_pad:
        padder = InputPadder(img0.shape, 16)
        img0, img1 = padder.pad(img0, img1)

    embt = torch.tensor(1 / 2).view(1, 1, 1, 1).float().to(device)

    imgt_pred = model(img0, img1, embt)
    imgt_pred = padder.unpad(imgt_pred)

    return Image.fromarray(
        (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    )
