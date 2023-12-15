import math
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, ParameterTuple

from src.utils.lr_schedule import dynamic_lr
from src.models.focal_softmax import FocalSoftmaxLoss
from src.models.lovasz_softmax import Lovasz_softmax


class CustomMutiLoss(nn.Cell):
    """自定义多标签损失函数"""

    def __init__(self, settings, cls_weight):
        super(CustomMutiLoss, self).__init__()
        self.settings = settings
        self.n_classes = self.settings.nclasses
        self.tau = self.settings.tau
        self.lambda_ = self.settings.lambda_
        self.gamma = self.settings.gamma
        self.class_log = math.log(self.settings.nclasses)

        self.lovasz = Lovasz_softmax()
        self.kl_loss = ops.KLDivLoss(reduction="none")

        self.log = ops.Log()
        self.greaterEqual = ops.GreaterEqual()
        self.greater = ops.Greater()
        self.less = ops.Less()
        self.expandDims = ops.ExpandDims()

        if self.settings.dataset == "SemanticKitti":
            alpha = np.log(1 + cls_weight)
            alpha = alpha / alpha.max()
        alpha[0] = 0

        self.focal_loss = FocalSoftmaxLoss(
            self.settings.nclasses, gamma=2, alpha_para=alpha, softmax=False)

    def construct(self, lidar_pred, camera_pred, label, label_mask):
        # compute pcd entropy
        lidar_pred_log = self.log(ops.clip_by_value(lidar_pred,
                                                    Tensor(1e-8, mindspore.float32),
                                                    Tensor(1, mindspore.float32)))
        # compute pcd entropy: p * log p
        pcd_entropy = -(lidar_pred * lidar_pred_log).sum(1) / self.class_log
        # compute img entropy
        camera_pred_log = self.log(ops.clip_by_value(camera_pred,
                                                     Tensor(1e-8, mindspore.float32),
                                                     Tensor(1, mindspore.float32)))

        loss_foc = self.focal_loss(lidar_pred, label, mask=label_mask)
        # loss_lov = self.lovasz(lidar_pred, label)
        loss_lov = 0

        # normalize to [0,1)
        img_entropy = -(camera_pred * camera_pred_log).sum(1) / self.class_log

        loss_foc_cam = self.focal_loss(camera_pred, label, mask=label_mask)
        # loss_lov_cam = self.lovasz(camera_pred, label)
        loss_lov_cam = 0

        pcd_confidence = 1. - pcd_entropy
        img_confidence = 1. - img_entropy
        information_importance = pcd_confidence - img_confidence
        pcd_guide_mask = self.greaterEqual(pcd_confidence, self.tau).astype(mindspore.float32)
        img_guide_mask = self.greaterEqual(img_confidence, self.tau).astype(mindspore.float32)

        pcd_guide_weight = self.greater(information_importance, 0.).astype(
            mindspore.float32) * information_importance.abs() * pcd_guide_mask
        img_guide_weight = self.less(information_importance, 0.).astype(
            mindspore.float32) * information_importance.abs() * img_guide_mask

        # compute kl loss
        loss_per_pcd = (self.kl_loss(
            lidar_pred_log, camera_pred) * self.expandDims(img_guide_weight, 1)).mean()
        loss_per_img = (self.kl_loss(
            camera_pred_log, lidar_pred) * self.expandDims(pcd_guide_weight, 1)).mean()
        loss_per = loss_per_pcd + loss_per_img

        total_loss = loss_foc + loss_lov * self.lambda_ + \
                     loss_foc_cam + loss_lov_cam * self.lambda_ + \
                     loss_per * self.gamma
        '''total_loss = loss_foc + loss_foc_cam + loss_per * self.gamma'''
        
        return total_loss


class CustomWithLossCell(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, settings, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, pcd_feature, img_feature, label_mask, input_label):

        lidar_pred, camera_pred = self._backbone(pcd_feature, img_feature)  # 前向计算得到网络输出

        return self._loss_fn(lidar_pred, camera_pred, input_label, label_mask)  # 得到多损失值


class CustomMitiMomentum(nn.Optimizer):
    """自定义定义多优化器"""

    def __init__(self, adam_params, sgd_encoder_params, sgd_decoder_params, learning_rate, momentum,
                 weight_decay, data_loader_len, n_epochs, warmup_epochs):
        group_params = [{'params': adam_params},
                        {'params': sgd_encoder_params},
                        {'params': sgd_decoder_params}]
        super(CustomMitiMomentum, self).__init__(learning_rate, group_params)
        self.momentum = Tensor(momentum)
        self.weight_decay = weight_decay
        self.adam_m = ParameterTuple(self.parameters[:246]).clone(prefix='adam_m', init='zeros')
        self.adam_v = ParameterTuple(self.parameters[:246]).clone(prefix='adam_v', init='zeros')
        self.sgd_acc = ParameterTuple(self.parameters[246:]).clone(prefix='sgd_acc', init='zeros')
        self.sgd_stat = ParameterTuple(self.parameters[246:]).clone(prefix='sgd_stat', init='ones')

        self.dy_lr = dynamic_lr(learning_rate, data_loader_len * warmup_epochs, data_loader_len * n_epochs)
        self.dy_lr_index = Parameter(0)

        self.adam_opt = ops.AdamWeightDecay()
        self.sgd_opt = ops.SGD(weight_decay=weight_decay, nesterov=True)

    def construct(self, gradients):
        """在训练中自动传入梯度gradients"""
        lr = Tensor(self.dy_lr[self.dy_lr_index])

        params = self.parameters  # 待更新的权重参数

        for p, m, v, grad in zip(params[:246], self.adam_m, self.adam_v, gradients[:246]):
            self.adam_opt(p, m, v, lr, 0.9, 0.999, 1e-08, 0.01, grad)

        for p, acc, sta, grad in zip(params[246:], self.sgd_acc, self.sgd_stat, gradients[246:]):
            self.sgd_opt(p, grad, lr, acc, self.momentum, sta)

        self.dy_lr_index += 1

        return params







