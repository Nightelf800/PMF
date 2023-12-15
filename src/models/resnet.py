from typing import Any, Type, Union, List, Optional

from mindspore import nn


class ConvNormActivation(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.BatchNorm2d.
        activation (nn.Cell, optional): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> conv = ConvNormActivation(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU
                 ) -> None:
        super(ConvNormActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                pad_mode='pad',
                padding=padding,
                group=groups
            )
        ]

        if norm:
            layers.append(norm(out_planes))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class ResidualBlockBase(nn.Cell):
    """
    ResNet residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_with (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super(ResidualBlockBase, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        assert group != 1 or base_width == 64, "ResidualBlockBase only supports groups=1 and base_width=64"
        self.conv1 = ConvNormActivation(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            norm=norm)
        self.conv2 = ConvNormActivation(
            out_channel,
            out_channel,
            kernel_size=3,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Type[Union[ResidualBlockBase, ResidualBlock]]): Block for network.
        layer_nums (List[int]): Numbers of block in different layers.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock, [3, 4, 6, 3])
    """

    def __init__(self,
                 block: Type[Union[ResidualBlockBase]],
                 layer_nums: List[int],
                 group: int = 1,
                 base_with: int = 64,
                 norm: Optional[nn.Cell] = None
                 ) -> None:
        super(ResNet, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        self.norm = norm
        self.in_channels = 64
        self.group = group
        self.base_with = base_with
        self.conv1 = ConvNormActivation(
            3, self.in_channels, kernel_size=7, stride=2, norm=norm)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2)

    def _make_layer(self,
                    block: Type[Union[ResidualBlockBase]],
                    channel: int,
                    block_nums: int,
                    stride: int = 1
                    ):
        """Block layers."""
        down_sample = None

        if stride != 1 or self.in_channels != self.in_channels * block.expansion:
            down_sample = ConvNormActivation(
                self.in_channels,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                norm=self.norm,
                activation=None)
        layers = []
        layers.append(
            block(
                self.in_channels,
                channel,
                stride=stride,
                down_sample=down_sample,
                group=self.group,
                base_width=self.base_with,
                norm=self.norm
            )
        )
        self.in_channels = channel * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.in_channels,
                    channel,
                    group=self.group,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(arch: str,
            block: Type[Union[ResidualBlockBase]],
            layers: List[int],
            num_classes: int,
            pretrained: bool,
            input_channel: int,
            **kwargs: Any
            ) -> ResNet:
    """ResNet architecture."""
    model = ResNet(block, layers, **kwargs)

    return model

def resnet34(
        num_classes: int = 1000,
        pretrained: bool = False,
        **kwargs: Any) -> ResNet:
    """
    ResNet34 architecture.

    Args:
        num_classes (int): Number of classification. Default: 1000.
        pretrained (bool): Download and load the pre-trained model. Default: False.

    Returns:
        ResNet

    Examples:
        >>> resnet34(num_classes=10, pretrained=True, **kwargs)
    """
    return _resnet(
        "resnet34", ResidualBlockBase, [
            3, 4, 6, 3], num_classes, pretrained, 512, **kwargs)