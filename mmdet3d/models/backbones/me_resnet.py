import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from mmdet.models import BACKBONES


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, n_outs):
        super(ResNetBase, self).__init__()
        self.n_outs = n_outs
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        if n_outs > 1:
            self.layer2 = self._make_layer(
                self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
            )
        if n_outs > 2:
            self.layer3 = self._make_layer(
                self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
            )
        if n_outs > 3:
            self.layer4 = self._make_layer(
                self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=3,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=3))

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        outs = []
        x = self.conv1(x)
        x = self.layer1(x)
        outs.append(x)
        if self.n_outs == 1:
            return outs
        x = self.layer2(x)
        outs.append(x)
        if self.n_outs == 2:
            return outs
        x = self.layer3(x)
        outs.append(x)
        if self.n_outs == 3:
            return outs
        x = self.layer4(x)
        outs.append(x)
        return outs


@BACKBONES.register_module()
class MEResNet3D(ResNetBase):
    def __init__(self, in_channels, depth, n_outs=4):
        if depth == 14:
            self.BLOCK = BasicBlock
            self.LAYERS = (1, 1, 1, 1)
        elif depth == 18:
            self.BLOCK = BasicBlock
            self.LAYERS = (2, 2, 2, 2)
        elif depth == 34:
            self.BLOCK = BasicBlock
            self.LAYERS = (3, 4, 6, 3)
        elif depth == 50:
            self.BLOCK = Bottleneck
            self.LAYERS = (4, 3, 6, 3)
        elif depth == 101:
            self.BLOCK = Bottleneck
            self.LAYERS = (3, 4, 23, 3)
        else:
            raise ValueError(f'invalid depth={depth}')

        super(MEResNet3D, self).__init__(in_channels, n_outs)
