"""

MIT License

Copyright (c) 2018 Maxim Berman
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""

from mindspore import Parameter, Tensor
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


class Lovasz_softmax(nn.Cell):
    def __init__(self, classes='present', per_image=False):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.gatherD = ops.GatherD()

    def construct(self, probas, labels):
        return self.lovasz_softmax(probas, labels, self.classes, self.per_image)

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def nonzero(self, input):
        """Return indexes of all nonzero/True elements.
        """
        input_np = input.asnumpy()
        output = input_np.nonzero()

        return Tensor.from_numpy(output[0])

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - ops.CumSum()(gt_sorted.astype(mindspore.float32), 0)
        union = gts + ops.CumSum()((1 - gt_sorted).astype(mindspore.float32), 0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_grad2(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = gt_sorted.shape[0]
        gts = gt_sorted.sum(0)
        intersection = gts - ops.CumSum()(gt_sorted.astype(mindspore.float32), 0)
        union = gts + ops.CumSum()((1 - gt_sorted).astype(mindspore.float32), 0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax(self, probas, labels, classes='present', per_image=False):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if per_image:
            loss = self.mean(self.lovasz_softmax_flat2(*self.flatten_probas(ops.ExpandDims()(prob, 0), ops.ExpandDims()(lab, 0)), classes=classes) for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat2(*self.flatten_probas(probas, labels), classes=classes)
        return loss

    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if ops.Size()(probas) == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.

        if probas.ndim == 1:
            probas = ops.ExpandDims()(probas, 0)
        C = probas.shape[1]
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).astype(mindspore.float32)  # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (ops.stop_gradient(fg) - class_pred).astype('float16').abs()
            errors_sorted, perm = ops.Sort(0, descending=True)(errors)
            perm = ops.stop_gradient(perm)
            fg_sorted = fg[perm]
            errors_sorted = errors_sorted.astype("float32")
            losses.append(ops.tensor_dot(errors_sorted, ops.stop_gradient(self.lovasz_grad(fg_sorted)), 1))
        return self.mean(losses)

    def lovasz_softmax_flat2(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if ops.Size()(probas) == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.

        if probas.ndim == 1:
            probas = ops.ExpandDims()(probas, 0)
        C = probas.shape[1]

        labels = ops.ExpandDims()(labels, 1)
        labels = ops.BroadcastTo((labels.shape[0], C))(labels)
        c = ops.ExpandDims()(Tensor(np.arange(C)), 0)
        c = ops.BroadcastTo((labels.shape[0], C))(c)

        fg = ops.Equal()(labels, c).astype(mindspore.float32)
        fg_nonzero = self.nonzero(fg.sum(0))
        # fg_nonzero = fg.sum(0).nonzero().squeeze()
        errors = (ops.stop_gradient(fg) - probas).astype('float16').abs()
        errors_sorted, perm = ops.Sort(0, descending=True)(errors)
        perm = ops.stop_gradient(perm)
        fg_sorted = self.gatherD(fg, 0, perm)
        errors_sorted = errors_sorted.astype('float32')
        dot = ops.tensor_dot(errors_sorted, ops.stop_gradient(self.lovasz_grad2(fg_sorted)), (0, 0))
        loss = dot.diagonal()
        loss = loss[fg_nonzero]
        loss_final = loss.mean()
        return loss_final

    def flatten_probas(self, probas, labels):
        """
        Flattens predictions in the batch
        """
        if probas.ndim == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.shape
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.shape
        probas = ops.Transpose()(probas, (0, 2, 3, 1)).view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        valid = self.nonzero(labels != 0)
        # valid = (labels != 0).astype("int32")
        # valid = valid.reshape((valid.shape[1], valid.shape[0])).squeeze()
        vprobas = probas[valid]
        vlabels = labels[valid]
        return vprobas, vlabels
