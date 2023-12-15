import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore import nn, ops

class CustomWithEvalCell(nn.Cell):

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = ops.Argmax(1)

    def construct(self, pcd_feature, img_feature, label_mask, input_label):

        lidar_pred, camera_pred = self.network(pcd_feature, img_feature)

        argmax = self.argmax(lidar_pred)
        # argmax_img = ops.Argmax(1)(camera_pred)

        return argmax, input_label


class IOUEval(nn.Metric):
    def __init__(self, n_classes, recorder, ignore=None, is_distributed=False):
        super(IOUEval, self).__init__()
        self.n_classes = n_classes
        self.recorder = recorder
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = Tensor(ignore).astype(mindspore.int64)
        self.is_distributed = is_distributed
        self.include = Tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).astype(mindspore.int64)
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.conf_matrix = ops.Zeros()((self.n_classes, self.n_classes), mindspore.float32)
        self.ones = Tensor(0)
        self.best_iou = 0.
        self.last_scan_size = 0

        self.zeros = ops.Zeros()
        self.ones_ops = ops.Ones()
        self.concat = ops.Concat(1)
        self.scatterNdAdd = ops.ScatterNdAdd()
        self.allReduce = ops.AllReduce()
        self.reduceSum = ops.ReduceSum()

    def clear(self):
        self.conf_matrix = self.zeros((self.n_classes, self.n_classes), mindspore.float32)
        self.ones = Tensor(0)
        self.last_scan_size = 0  # for when variable scan size is used

    def update(self, *inputs):  # x=preds, y=targets

        if len(inputs) != 2:
            raise ValueError('Distribute accuracy needs 2 input (y_correct), but got {}'.format(len(inputs)))

        x = inputs[0]
        y = inputs[1]
        if isinstance(x, np.ndarray):
            x = Tensor.from_numpy(np.array(x)).astype(mindspore.int64)
        if isinstance(y, np.ndarray):
            y = Tensor.from_numpy(np.array(y)).astype(mindspore.int64)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1, 1)  # de-batchify

        y_row = y.reshape(-1, 1)  # de-batchify

        # idxs are labels and predictions
        idxs = self.concat((x_row, y_row))

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[0]:
            self.ones = self.ones_ops((idxs.shape[0]), mindspore.float32)
            self.last_scan_size = idxs.shape[0]

        # make confusion matrix (cols = gt, rows = pred)
        # GPU
        self.conf_matrix = self.scatterNdAdd(self.conf_matrix, idxs, self.ones)


    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.copy().astype(mindspore.float32)
        if self.is_distributed:
            conf = self.allReduce(conf)
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = self.reduceSum(conf, 1) - tp
        fn = self.reduceSum(conf, 0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES


    def eval(self):
        """计算最终评估结果"""
        # print("-------------输出IOU-----------")
        iou_mean, iou = self.getIoU()
        iou_mean_float = float(iou_mean)
        log_str = ">>> {} mIOU[{:.4f}] BestmIOU[{:.4f}]".format("Validation", iou_mean_float, self.best_iou)
        if context.get_context('device_target') == 'GPU':
            if self.recorder is not None:
                self.recorder.logger.info(log_str)
        else:
            print(log_str)
        
        if self.best_iou < iou_mean_float:
            self.best_iou = iou_mean_float
        self.clear()
        return iou_mean_float
