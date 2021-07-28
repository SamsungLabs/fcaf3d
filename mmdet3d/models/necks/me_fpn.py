from torch import nn
import MinkowskiEngine as ME

from mmdet.models import NECKS


@NECKS.register_module()
class MEFPN3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MEFPN3D, self).__init__()
        self.n_scales = len(in_channels)
        for i in range(self.n_scales):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

    def forward(self, inputs):
        outs = []
        x = inputs[-1]
        for i in range(self.n_scales - 1, -1, -1):
            if i < self.n_scales - 1:
                x = inputs[i] + x
            outs.append(self.__getattr__(f'out_block_{i}')(x))
            if i > 0:
                x = self.__getattr__(f'up_block_{i}')(x)
        return outs[::-1]

    def init_weights(self):
        pass

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )


@NECKS.register_module()
class MEFPN3DV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MEFPN3DV2, self).__init__()
        self.n_scales = len(in_channels)
        for i in range(self.n_scales):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(out_channels))
            self.__setattr__(f'lateral_block_{i}', self._make_block(in_channels[i], out_channels))

    def forward(self, inputs):
        outs = []
        x = self.__getattr__(f'lateral_block_{self.n_scales - 1}')(inputs[-1])
        for i in range(self.n_scales - 1, -1, -1):
            if i < self.n_scales - 1:
                x = self.__getattr__(f'lateral_block_{i}')(inputs[i]) + x
            outs.append(x)
            if i > 0:
                x = self.__getattr__(f'up_block_{i}')(x)
        return outs[::-1]

    def init_weights(self):
        pass

    @staticmethod
    def _make_up_block(n_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                n_channels,
                n_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(n_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(n_channels, n_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(n_channels),
            ME.MinkowskiReLU()
        )

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU()
        )
