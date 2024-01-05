"""
# -*- coding: utf-8 -*-
# @Author  : youngx
# @Time    : 2022/6/27 15:56
# @FileName : Convert_sim_onnx
"""
import torch
import onnx
from onnxsim import simplify
from darknet19 import DarkNet
import dataloder


def Convert2onnx(ptfile):
    class_num = len(dataloder.TYPE) + len(dataloder.COLOR)

    net = DarkNet(class_num=class_num)
    net.to("cpu")
    net.load_state_dict(torch.load(ptfile, map_location="cpu"))

    # # model convert to onnx
    a = torch.randn((1, 3, 448, 448))
    input_node = ["inputNode"]
    out_node = ["outPutNode"]
    net.eval()
    saveOnnxName = "multi_label_classification.onnx"
    torch.onnx.export(net, a, saveOnnxName, verbose=True, training=torch.onnx.TrainingMode.EVAL, input_names=input_node,
                      output_names=out_node, opset_version=9)

    return saveOnnxName


def SimOnnx(onnxFile):
    simFile = onnxFile.replace(".onnx", "_sim.onnx")
    model = onnx.load(onnxFile)

    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simFile)
    print('finished exporting onnx')


if __name__ == '__main__':
    ptfile = "best.pt"
    onnx_file = Convert2onnx(ptfile)

    SimOnnx(onnx_file)
