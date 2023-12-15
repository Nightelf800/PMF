import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import context, load_checkpoint, load_param_into_net
from src.models.resnet import resnet34
# from mindvision.classification.models import resnet34
from .salsanext import SalsaNext


class ResidualBasedFusionBlock(nn.Cell):
    def __init__(self, pcd_channels, img_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.fuse_conv = nn.SequentialCell(
            nn.Conv2d(pcd_channels + img_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1, pad_mode='pad', has_bias=True),
            nn.LeakyReLU(alpha=0.01),
            nn.BatchNorm2d(pcd_channels)
        )

        self.attention = nn.SequentialCell(
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(pcd_channels),
            nn.ReLU(),
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(pcd_channels),
            nn.Sigmoid()
        )

    def construct(self, pcd_feature, img_feature):
        cat_feature = ops.Concat(1)((pcd_feature, img_feature))
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        out = fuse_out * attention_map + pcd_feature
        return out


# -----------------------------------------------------------------------
class ResNet(nn.Cell):
    def __init__(self, in_channels=3, backbone="resnet50", dropout_rate=0.2,
                 pretrained=True, pretrained_path=""):
        super(ResNet, self).__init__()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

        if backbone == "resnet34":
            if pretrained:
                param_dict = load_checkpoint(pretrained_path)
                net = resnet34()
                load_param_into_net(net, param_dict)
            else:
                net = resnet34()
            self.expansion = 1
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))

        self.feature_channels = [64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]
        self.backbone_name = backbone

        # Note that we do not downsample for conv1
        # self.conv1 = net.conv1
        conv2d_weight = mindspore.Tensor(net.conv1.features[0].weight.data)
        if in_channels == 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                   stride=1, padding=3, pad_mode='pad',
                                   weight_init=conv2d_weight)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=1, padding=3, pad_mode='pad')
        self.bn1 = net.conv1.features[1]
        self.relu = net.conv1.features[2]
        self.maxpool = net.max_pool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # dropout
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)

    def construct(self, x, img_feature=[]):
        # pad input to be divisible by 16 = 2 ** 4
        h, w = x.shape[2], x.shape[3]
        # check input size
        if h % 16 != 0 or w % 16 != 0:
            if not False:
                raise AssertionError("invalid input size: {}".format(x.shape))

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        # inter_features = []
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        conv1_pool_out = self.maxpool(conv1_out)
        layer1_out = self.layer1(conv1_pool_out)
        layer2_out = self.layer2(layer1_out)  # downsample
        layer3_out = self.dropout(self.layer3(layer2_out))  # downsample
        layer4_out = self.dropout(self.layer4(layer3_out))  # downsample

        return [layer1_out, layer2_out, layer3_out, layer4_out]


class ASPP(nn.Cell):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # self.mean = ops.AdaptiveAvgPool2D((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1, pad_mode='valid', has_bias=True)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1, pad_mode='valid', has_bias=True)
        self.atrous_block6 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=6, pad_mode='pad', dilation=6, has_bias=True)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, pad_mode='pad', dilation=12, has_bias=True)
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, pad_mode='pad', dilation=18, has_bias=True)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1, pad_mode='valid', has_bias=True)

    def construct(self, x):
        size = x.shape[2:]

        # image_features = self.mean(x)
        image_features = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]), stride=1)(x)
        image_features = self.conv(image_features)
        image_features = nn.ResizeBilinear()(image_features, size=size)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(ops.Concat(1)([
            image_features, atrous_block1, atrous_block6,
            atrous_block12, atrous_block18]))
        return net


class SalsaNextFusion(SalsaNext):
    def __init__(self, in_channels=8, nclasses=20, base_channels=32, img_feature_channels=[]):
        super(SalsaNextFusion, self).__init__(in_channels=in_channels, base_channels=base_channels,
                                              nclasses=nclasses, softmax=True)

        self.fusionblock_1 = ResidualBasedFusionBlock(self.base_channels * 2, img_feature_channels[0])
        self.fusionblock_2 = ResidualBasedFusionBlock(self.base_channels * 4, img_feature_channels[1])
        self.fusionblock_3 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[2])
        self.fusionblock_4 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[3])

        self.aspp = ASPP(self.base_channels * 8, self.base_channels * 8)

    def construct(self, x, img_feature=[]):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down0c = self.fusionblock_1(down0c, img_feature[0])

        down1c, down1b = self.resBlock2(down0c)
        down1c = self.fusionblock_2(down1c, img_feature[1])

        down2c, down2b = self.resBlock3(down1c)
        down2c = self.fusionblock_3(down2c, img_feature[2])

        down3c, down3b = self.resBlock4(down2c)
        down3c = self.fusionblock_4(down3c, img_feature[3])

        down5c = self.aspp(self.resBlock5(down3c))

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)
        if self.softmax:
            logits = nn.Softmax(1)(logits)

        return logits


class RGBDecoder(nn.Cell):
    def __init__(self, in_channels=[], nclasses=4, base_channels=64):
        super(RGBDecoder, self).__init__()

        self.up_4a = nn.SequentialCell(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1, pad_mode='pad',
                      has_bias=True),
            nn.LeakyReLU(alpha=0.01),
            nn.BatchNorm2d(base_channels),
        )
        self.up_3a = nn.SequentialCell(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1, pad_mode='pad',
                      has_bias=True),
            nn.LeakyReLU(alpha=0.01),
            nn.BatchNorm2d(base_channels),
        )
        self.up_2a = nn.SequentialCell(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1, pad_mode='pad',
                      has_bias=True),
            nn.LeakyReLU(alpha=0.01),
            nn.BatchNorm2d(base_channels),
        )
        self.up_1a = nn.SequentialCell(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 1, pad_mode='valid',
                      has_bias=True),
            nn.LeakyReLU(alpha=0.01),
            nn.BatchNorm2d(base_channels),
        )
        self.conv = nn.Conv2d(base_channels, nclasses, kernel_size=3, padding=1, pad_mode='pad',
                              has_bias=True)

        self.resizeBilinear1 = nn.ResizeBilinear()
        self.resizeBilinear2 = nn.ResizeBilinear()
        self.resizeBilinear3 = nn.ResizeBilinear()
        self.resizeBilinear4 = nn.ResizeBilinear()

    def construct(self, inputs):
        up_4a = self.up_4a(inputs[3])
        up_4a = self.resizeBilinear1(up_4a, scale_factor=2)
        up_3a = self.up_3a(ops.Concat(1)((up_4a, inputs[2])))
        up_3a = self.resizeBilinear2(up_3a, scale_factor=2)
        up_2a = self.up_2a(ops.Concat(1)((up_3a, inputs[1])))
        up_2a = self.resizeBilinear3(up_2a, scale_factor=2)
        up_1a = self.up_1a(ops.Concat(1)((up_2a, inputs[0])))
        up_1a = self.resizeBilinear4(up_1a, scale_factor=2)
        out = self.conv(up_1a)
        out = nn.Softmax(1)(out)
        return out



class PMFNet(nn.Cell):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, base_channels=32,
                 imagenet_pretrained=True, pretrained_path="", image_backbone="resnet34"):
        super(PMFNet, self).__init__()

        self.camera_stream_encoder = ResNet(
            in_channels=img_channels,
            pretrained=imagenet_pretrained,
            pretrained_path=pretrained_path,
            backbone=image_backbone)

        self.camera_stream_decoder = RGBDecoder(
            self.camera_stream_encoder.feature_channels,
            nclasses=nclasses, base_channels=self.camera_stream_encoder.expansion * 16)

        self.lidar_stream = SalsaNextFusion(
            in_channels=pcd_channels, nclasses=nclasses, base_channels=base_channels,
            img_feature_channels=self.camera_stream_encoder.feature_channels)

    def construct(self, pcd_feature, img_feature):
        img_feature = self.camera_stream_encoder(img_feature)
        lidar_pred = self.lidar_stream(pcd_feature, img_feature)
        camera_pred = self.camera_stream_decoder(img_feature)

        return lidar_pred, camera_pred
