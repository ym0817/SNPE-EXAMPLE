"""
.pth -> .onnx -> sim_onnx -> ncnn
"""

import os
import time

import imageio

from darknet19 import DarkNet
from darknet53 import darknet53
from dataloder import ImageLoader
from torch.utils.data import DataLoader
import dataloder
import torch
import numpy as np
import cv2
import onnxsim
import onnx
# from onnxsim import simplify
import ncnn


class modelConvert():
    def __init__(self, weightPath):
        self.weightPath = weightPath
        device = "cuda"
        class_num = len(dataloder.TYPE) + len(dataloder.COLOR)
        self.net = DarkNet(class_num)
        # self.net.load_state_dict(torch.load(self.weightPath))
        self.outDir = "convert_model"

    def convert2Onnx(self):
        out_name = "multi_class_classification.onnx"
        netInput = torch.randn(1, 3, 448, 448)
        inputName = ["Input_Image"]
        outputName = ["Predict"]
        self.outFile = os.path.join(self.outDir, out_name)
        torch.onnx.export(self.net, netInput, self.outFile,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=False,
                          input_names=inputName,
                          output_names=outputName,
                          # export_params=False,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          verbose=False,
                          dynamic_axes=None,
                          # export_params=False,
                          opset_version=12)

        model_onnx = onnx.load(self.outFile)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        print("onnx sim")
        # onnx model simplify
        sim_out_name = "multi_class_classification_sim.onnx"
        onnx_model = onnx.load(self.outFile)  # load onnx model
        model_simp, check = onnxsim.simplify(onnx_model,
                                             dynamic_input_shape=False,
                                             input_shapes=None)

        sim_out_name = os.path.join(self.outDir, sim_out_name)
        onnx.save(model_simp, sim_out_name)

    def convert2Ncnn(self):
        onnx2ncnn = r"D:\MyNAS\ncnn-20220216-windows-vs2017\x64\bin\onnx2ncnn.exe"
        outParamName = "multi_class_classification.param"
        outBinName = "multi_class_classification.bin"
        ncnnParam = os.path.join(self.outDir, outParamName)
        ncnnBin = os.path.join(self.outDir, outBinName)

        r_v = os.system(onnx2ncnn + " " + self.outFile + " " + ncnnParam + " " + ncnnBin)
        print(r_v)


class modelEval():
    def __init__(self, target_size=448):
        self.target_size = target_size

    def onnxEval(self, savedir, onnxpath):
        net = cv2.dnn.readNetFromONNX(onnxpath)
        if net:
            print("load model")

        valdir = r"D:\MyNAS\Multi-label-classification-main\test_data"
        valset = ImageLoader(valdir)
        valLoader = DataLoader(valset, batch_size=1, shuffle=False)

        pred_num = 0
        for id, batch in enumerate(valLoader):
            _, label, imgfile = batch[0].float(), batch[1].float(), batch[2]
            label = label.squeeze(1)
            color_gt = label[0, :len(dataloder.COLOR)]
            type_gt = label[0, len(dataloder.COLOR):]

            _, color_gt_index = torch.max(color_gt, 0)
            _, type_gt_index = torch.max(type_gt, 0)

            imagename = "{}_{}_{}.png".format(dataloder.COLOR[int(color_gt_index)], dataloder.TYPE[int(type_gt_index)],
                                              str(id))

            img = cv2.imread(imgfile[0])
            img = img[..., :3]
            input = cv2.dnn.blobFromImage(img, 1/255.0, (self.target_size, self.target_size), (0,0,0))
            startTime = time.time()
            net.setInput(input)
            outpred = net.forward("Predict")

            color_pred = outpred[0, :len(dataloder.COLOR)]
            type_pred = outpred[0, len(dataloder.COLOR):]

            color_cls_index = np.argmax(color_pred, 0)
            color_max_prob = color_pred[color_cls_index]

            type_cls_index = np.argmax(type_pred, 0)
            type_max_prob = type_pred[type_cls_index]
            if color_cls_index == color_gt_index.item() and type_cls_index.item() == type_gt_index:
                pred_num += 1
            color_mess = '%s : %.3f' % (dataloder.COLOR[int(color_cls_index)], color_max_prob)
            cv2.putText(img, color_mess, (int(10), int(100 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

            type_mess = '%s : %.3f' % (dataloder.TYPE[int(type_cls_index)], type_max_prob)
            cv2.putText(img, type_mess, (int(10), int(200 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

            cv2.imwrite(savedir + "/" + imagename, img)
            endTime = time.time()
            print("%s use time %.3f" % (os.path.basename(imgfile[0]), endTime - startTime))

        print("eval acc : %.3f , total num %d,  %d" % (pred_num / len(valLoader), pred_num, len(valLoader)))


def ncnnEval(self, ncnnParam, ncnnBin, saveDir):
    self.ncnnNet = ncnn.Net()
    self.ncnnNet.load_param(ncnnParam)
    self.ncnnNet.load_model(ncnnBin)

    pred_num = 0
    startTime0 = time.time()
    for id, batch in enumerate(self.valLoader):
        startTime = time.time()
        _, label, imgfile = batch[0].float().to(), batch[1].float().to(), batch[2]
        label = label.squeeze(1)
        color_gt = label[0, :len(dataloder.COLOR)]
        type_gt = label[0, len(dataloder.COLOR):]
        _, color_gt_index = torch.max(color_gt, 0)
        _, type_gt_index = torch.max(type_gt, 0)
        imagename = "{}_{}_{}.png".format(dataloder.COLOR[int(color_gt_index)], dataloder.TYPE[int(type_gt_index)],
                                          str(id))

        img0 = cv2.imread(imgfile[0])
        # img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = img0
        # img_h = img.shape[0]
        # img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            self.target_size,
            self.target_size,
        )
        mat_in.substract_mean_normalize([0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        # mat_in.substract_mean_normalize([255.0,255.0,255.0], [])
        ex = self.ncnnNet.create_extractor()
        ex.input("Input_Image", mat_in)

        _, mat_out = ex.extract("Predict")
        pred = np.array(mat_out).reshape(1, -1)
        color_pred = pred[0, :len(dataloder.COLOR)]
        type_pred = pred[0, len(dataloder.COLOR):]

        color_cls_index = np.argmax(color_pred, 0)
        color_max_prob = color_pred[color_cls_index]

        type_cls_index = np.argmax(type_pred, 0)
        type_max_prob = type_pred[type_cls_index]

        if color_cls_index == color_gt_index.item() and type_cls_index == type_gt_index.item():
            pred_num += 1
        color_mess = '%s : %.3f' % (dataloder.COLOR[int(color_cls_index)], color_max_prob)
        cv2.putText(img0, color_mess, (int(10), int(100 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        type_mess = '%s : %.3f' % (dataloder.TYPE[int(type_cls_index)], type_max_prob)
        cv2.putText(img0, type_mess, (int(10), int(200 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imwrite(saveDir + "/" + imagename, img0)
        endTime = time.time()
        print("%s use time %.3f" % (os.path.basename(imgfile[0]), endTime - startTime))

    print("eval acc : %.3f , total num %d,  %d" % (pred_num / len(self.valLoader), pred_num, len(self.valLoader)))
    startTime2 = time.time()
    print(startTime2 - startTime0)


if __name__ == '__main__':
    weightpath = "darkent19_best.pt"
    convertTools = modelConvert(weightpath)
    convertTools.convert2Onnx()
    #     convertTools.convert2Ncnn()

    #     ncnnParam = r"D:\MyNAS\multi_class\convert_model\multi_class_classification.param"
    #     ncnnBin = r"D:\MyNAS\multi_class\convert_model\multi_class_classification.bin"
    # saveDir = "./test_result/"
    # onnxPath = "./convert_model/multi_class_classification_sim.onnx"
    # evalTools = modelEval()
    # evalTools.onnxEval(saveDir, onnxPath)
#     evalTools.ncnnEval(ncnnParam, ncnnBin, saveDir)
