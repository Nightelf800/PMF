import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np


class FocalSoftmaxLoss(nn.Cell):
    def __init__(self, n_classes, gamma=1, alpha_para=0.8, softmax=True):
        super(FocalSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.n_classes = n_classes
        self.gatherD = ops.GatherD()
        self.log = ops.Log()
        self.pow = ops.Pow()
        self.softmax_col = ops.Softmax(1)

        if isinstance(alpha_para, list):
            if not len(alpha_para) == n_classes:
                raise AssertionError("len(alpha_para)!=n_classes: {} vs. {}".format(len(alpha_para), n_classes))
            self.alpha_para = Tensor(alpha_para)
        elif isinstance(alpha_para, np.ndarray):
            if not alpha_para.shape[0] == n_classes:
                raise AssertionError("len(alpha_para)!=n_classes: {} vs. {}".format(len(alpha_para), n_classes))
            self.alpha_para = Tensor.from_numpy(alpha_para)
        else:
            if not alpha_para < 1 and alpha_para > 0:
                raise AssertionError("invalid alpha_para: {}".format(alpha_para))
            self.alpha_para = ops.Zeros()(n_classes, mindspore.float32)
            self.alpha_para[0] = alpha_para
            self.alpha_para[1:] += (1-alpha_para)
        self.softmax = softmax

    def construct(self, x, target, mask):
        """compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        """

        if x.ndim > 2:
            pred = x.view(x.shape[0], x.shape[1], -1)
            pred = pred.transpose(0, 2, 1)
            pred = pred.view(-1, x.shape[1])
        else:
            pred = x

        target = target.view(-1, 1)


        if self.softmax:
            pred_softmax = ops.Softmax(1)(pred)
        else:
            pred_softmax = pred

        pred_softmax = self.gatherD(pred_softmax, 1, target.astype(mindspore.int32)).view(-1)
        pred_logsoft = self.log(ops.clip_by_value(pred_softmax,
                                                   Tensor(1e-6, mindspore.float32),
                                                   Tensor(1, mindspore.float32)))

        alpha_para = self.gatherD(self.alpha_para, 0, target.squeeze().astype(mindspore.int32))
        loss = - self.pow(1 - pred_softmax, self.gamma)
        loss = loss * pred_logsoft * alpha_para
        if mask.ndim > 1:
            mask = mask.view(-1)
        mask = mask.astype('float32')
        loss = (loss * mask).sum() / mask.sum()
        return loss


if __name__ == "__main__":
    criterion = FocalSoftmaxLoss(n_classes=10, gamma=1, alpha_para=0.8)
    target = np.arange(0, 10)
    print(target)
    test_input = ops.UniformReal()(10, 10)
    mask = ops.Ones(10)
    mask[4] = 0
    loss = criterion(test_input, target, mask)
    print(loss)