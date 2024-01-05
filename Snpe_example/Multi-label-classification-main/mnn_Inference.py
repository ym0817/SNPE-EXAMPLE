"""
# -*- coding: utf-8 -*-
# @Author  : youngx
# @Time    : 2022/6/27 16:08
# @FileName : mnn_Inference
"""
from __future__ import print_function
import os
import numpy as np
import MNN
import dataloder
from dataloder import ImageLoader
from torch.utils.data import DataLoader
import torch
import cv2
from tqdm import tqdm


def MNN_Inference(mnnFile, img, INPUTSIZE=448):
    interpreter = MNN.Interpreter(mnnFile)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    image = np.array(img)

    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    # image = image.transpose((2, 0, 1))

    # construct tensor from np.ndarray
    tmp_input = MNN.Tensor((1, 3, INPUTSIZE, INPUTSIZE), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)

    return output_tensor.getData()


def main(fold1, fold2, mnnFile):
    valset = ImageLoader(fold1)
    valLoader = DataLoader(valset, batch_size=1, shuffle=False)

    for id, batch in tqdm(enumerate(valLoader)):
        img, label, imgfile = batch[0].float(), batch[1].float(), batch[2]
        label = label.squeeze(1)
        color_gt = label[0, :len(dataloder.COLOR)]
        type_gt = label[0, len(dataloder.COLOR):]

        _, color_gt_index = torch.max(color_gt, 0)
        _, type_gt_index = torch.max(type_gt, 0)
        imagename = "{}_{}_{}.png".format(dataloder.COLOR[int(color_gt_index)], dataloder.TYPE[int(type_gt_index)],
                                          str(id))
        pred = MNN_Inference(mnnFile, img)
        color_pred = pred[:len(dataloder.COLOR)]
        type_pred = pred[len(dataloder.COLOR):]

        color_cls_index = np.argmax(color_pred)
        color_max_prob = color_pred[color_cls_index]

        type_cls_index = np.argmax(type_pred)
        type_max_prob = type_pred[type_cls_index]

        image = cv2.imread(imgfile[0])
        color_mess = '%s : %.3f' % (dataloder.COLOR[int(color_cls_index)], color_max_prob)
        cv2.putText(image, color_mess, (int(10), int(200 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

        type_mess = '%s : %.3f' % (dataloder.TYPE[int(type_cls_index)], type_max_prob)
        cv2.putText(image, type_mess, (int(10), int(300 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

        cv2.imwrite(fold2 + "/" + imagename, image)


if __name__ == '__main__':
    mnnFile = "MNN_Infer/multi_label_classification_sim.mnn"
    testDir = "eval_test/test_img"
    saveDir = "eval_test/test_result"

    main(testDir, saveDir, mnnFile)
