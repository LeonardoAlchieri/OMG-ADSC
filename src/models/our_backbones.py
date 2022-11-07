import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import Tensor, einsum, nn
from torchvision.models import convnext_base
from vit_pytorch.cross_vit import CrossViT


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # S-Former after stage3
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        num_patches=7 * 7,
        dim=256,
        depth=1,
        heads=8,
        mlp_dim=512,
        dim_head=32,
        dropout=0.0,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or"
                " a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.spatial_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        nf = x.size(1)

        x = x.contiguous().view(-1, 3, 112, 112)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([1, 64, 28, 28])
        x = self.layer1(x)  # torch.Size([1, 64, 28, 28])
        x = self.layer2(x)  # torch.Size([1, 128, 14, 14])
        x = self.layer3(x)  # torch.Size([1, 256, 7, 7])
        b_l, c, h, w = x.shape
        x = x.reshape((b_l, c, h * w))
        x = x.permute(0, 2, 1)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.spatial_transformer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape((b, c, h, w))
        x = self.layer4(x)  # torch.Size([1, 512, 4, 4])
        x = self.avgpool(x)
        x = x.view(bs, nf, -1)
        #        x = torch.flatten(x, 1)
        return x


def spatial_transformer(loading_device: str = "cuda"):
    print("Loading backbone")
    return ResNet(BasicBlock, [2, 2, 2, 2]).to(loading_device)


def convnext(loading_device: str = "cuda", pretrained: bool = False):
    print("Loading backbone")
    return convnext_base(pretrained=pretrained).to(loading_device)


def facebval(loading_device: str = 'cuda'):
    print("** Loading backbone")
    return FaceBVAL(n_blocks=10).to(loading_device)

def facebval_convnext(loading_device: str = 'cuda'):
    print("** Loading backbone")
    return FaceBVAL(n_blocks=10, base_model='convnext').to(loading_device)

def facebval_convnext_tiny(loading_device: str = 'cuda'):
    print("** Loading backbone")
    return FaceBVAL(n_blocks=10, base_model='convnext_tiny').to(loading_device)

def facebval_convnext_small(loading_device: str = 'cuda'):
    print("** Loading backbone")
    return FaceBVAL(n_blocks=10, base_model='convnext_small').to(loading_device)

def VICRegL(loading_device: str = 'cuda'):
    print("** Loading backbone")
    return torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p75')


class EmoCrossViT(nn.Module):
    def __init__(self):
        super(EmoCrossViT, self).__init__()
        self.s_former = CrossViT(
            image_size=256,
            num_classes=10,
            depth=4,  # number of multi-scale encoding blocks
            sm_dim=192,  # high res dimension
            sm_patch_size=16,  # high res patch size (should be smaller than lg_patch_size)
            sm_enc_depth=2,  # high res depth
            sm_enc_heads=8,  # high res heads
            sm_enc_mlp_dim=2048,  # high res feedforward dimension
            lg_dim=384,  # low res dimension
            lg_patch_size=64,  # low res patch size
            lg_enc_depth=3,  # low res depth
            lg_enc_heads=8,  # low res heads
            lg_enc_mlp_dim=2048,  # low res feedforward dimensions
            cross_attn_depth=2,  # cross attention rounds
            cross_attn_heads=8,  # cross attention heads
            dropout=0.5,
            emb_dropout=0.1,
        )

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.s_former(x)
        return x.view(b, f, -1)


class FaceBVAL(nn.Module):
    """Class for the FaceBVAL Neural Network, a modification of ResNet50 to be trained on AffectNet

    Args:
        n_blocks (int, optional): number of blocks for the final sequential mapping; defines
        the valence-arousal plane (thus, an `n_blocks` x `n_blocks` plane will be defined) . Defaults to 10.
    """

    def __init__(self, n_blocks: int = 10, base_model: str = 'resnet50'):
        super(FaceBVAL, self).__init__()
        if base_model == 'resnet50':
            self.features = nn.Sequential(*list(models.resnet50().children())[:-1])
            # self.reduction_layer = nn.Sequential(
            #     nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)
            # )
        elif base_model == 'convnext':
            self.features = nn.Sequential(*list(models.convnext_base(pretrained=True).children())[:-1])
            self.reduction_layer = nn.Sequential(
                nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)
            )
        elif base_model == 'convnext_tiny':
            self.features = nn.Sequential(*list(models.convnext_tiny(pretrained=True).children())[:-1])
            self.reduction_layer = nn.Sequential(
                nn.Linear(768, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)
            )
        elif base_model == 'convnext_small':
            self.features = nn.Sequential(*list(models.convnext_small(pretrained=True).children())[:-1])
            self.reduction_layer = nn.Sequential(
                nn.Linear(768, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)
            )
        else:
            raise ValueError(f'Base model {base_model} not supported')

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for the neural network. Takes the input data an produces different outputs.

        Args:
            x (Tensor): input tensor. Must be 4 dimensional,
            of type `(number_of_elements, channels, pixel_width, pixel_height)`

        Returns:
        **WARNING**: THIS IS THE COMMENT FOR THE OLD METHOD. NOW THE METHOD RETURNS JUST THE FACEBVAL EMBEDDING
            Tuple[Tensor, Tensor, Tensor, Tensor]: the model outputs 4 different values:
            - `x_emot`: the emotional value, i.e. multiclass classification (11 classes), of shape `(number_of_elements, 11)`
            - `x_val_m`: valence value, of shape `(number_of_elements,)`
            - `x_aro_m`: arousal value, of shape `(number_of_elements,)`
            - `map_valaro`: map of activation in the valence-arousal plane, of shape `(number_of_elements,n_blocks,n_blocks)`
        """
        bs = x.size(0)
        nf = x.size(1)
        x = x.contiguous().view(-1, 3, 112, 112)
        x = self.features(x)
        x = x.view(bs*nf, -1)
        # x = self.reduction_layer(x)
        return x