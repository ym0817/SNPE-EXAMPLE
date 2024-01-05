# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 13:56  2022-03-31
import sys

sys.path.append(r"D:\MyNAS\TensorRT\TensorRT-8.4.0.6.Windows10.x86_64.cuda-10.2.cudnn8.3\TensorRT-8.4.0.6\lib")
import imageio
import cv2
import argparse
import os

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


def Inference(engine, input_data, input_device, out_data, out_device, stream, repeat_time=1):
    """
    inference
    :param engine: trt engine
    :param input_data: input data
    :param input_device: input data device number on cuda
    :param out_data: output data  placeholder
    :param out_device: output data device number on cuda
    :param stream:
    :param repeat_time:
    :return:
    """
    # load random data to page-locked buffer
    assert len(input_data) == len(input_device)
    for i_data, d_mem in zip(input_data, input_device):
        cuda.memcpy_htod_async(d_mem, i_data, stream)

    for _ in range(repeat_time):
        # # 将输入数据放到对应开辟的gpu 上, htod
        for o_data, o_device in zip(out_data, input_device):
            cuda.memcpy_htod_async(o_device, o_data, stream)

        # # 输入和输出数据的内存号
        bindings = [int(x) for x in input_device] + [int(x) for x in out_device]
        engine.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # # 将网络结果 由开辟的GPU 放到CPU上 dtoh
        for o_data, o_device in zip(out_data, out_device):
            cuda.memcpy_dtoh_async(o_data, o_device, stream)
        stream.synchronize()


def build_engin(onxpath, engine_file_path):
    """
    model load and convert
    :param onxpath: onnx model path
    :param engine_file_path: if save trt model converted from onnx
    :return:
    """
    TRT_LOGGER = trt.Logger()
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if os.path.exists(engine_file_path):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("load onnx parse")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)

        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1

        if not os.path.exists(onxpath):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onxpath))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onxpath))
        with open(onxpath, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        input_shape = [network.get_input(i).shape for i in range(network.num_inputs)]
        outputs_shape = [network.get_output(i).shape for i in range(network.num_outputs)]

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onxpath))
        plan = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        assert plan, "build serialized network error"

        # # convert trt file
        if engine_file_path:
            with open(engine_file_path, "wb") as f:
                f.write(plan)

        # # return engine
        engine = runtime.deserialize_cuda_engine(plan)

        return engine


def allocate_buffers(engine):
    """
    allocate cuda buffer
    :param engine:
    :return:
    """
    inputs_data = list()
    inputs_mem = list()
    output_data = list()
    output_mem = list()
    bindings = list()
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # bindings 记录的是输入和输出在GPU 开辟的内存
        bindings.append(int(device_mem))

        # 记录开辟的输入和输出数据的大小和内存
        if engine.binding_is_input(binding):
            inputs_data.append(host_mem)
            inputs_mem.append(device_mem)
        else:
            output_data.append(host_mem)
            output_mem.append(device_mem)
    return inputs_data, inputs_mem, output_data, output_mem, stream


def main(onxpath, engine_file_path=""):
    engine = build_engin(onxpath, engine_file_path)
    context = engine.create_execution_context()
    inputs_data, inputs_mem, output_data, output_mem, stream = allocate_buffers(engine)
    import time
    s1 = time.time()
    test_image = "test_result/black_dress_0.png"
    data = imageio.imread(test_image)
    data = cv2.resize(data, (448, 448))
    data = data / 255.0
    data = np.ascontiguousarray(data.transpose(2, 0, 1))
    data = np.expand_dims(data, 0)
    data = np.array(data, dtype=np.float32, order='C')
    # s1 = time.time()
    inputs_data[0] = data

    # # 进行推理
    Inference(engine=context, input_data=inputs_data, input_device=inputs_mem,
              out_data=output_data, out_device=output_mem,
              stream=stream)

    output_data = output_data[0].reshape(1, 9)
    s2 = time.time()
    print(s2-s1)


if __name__ == '__main__':
    onnx_path = "yolov5s.onnx"
    onnx_path = "./convert_model/multi_class_classification_sim.onnx"
    trt_path = onnx_path.replace(".onnx", ".trt")
    main(onnx_path)
