# Multi-label-classification

### 2022.6 
加入MNN 和TNN 推理代码
```python
import MNN
def MNN_Inference(mnnFile, img, INPUTSIZE=448):
    interpreter = MNN.Interpreter(mnnFile)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    image = np.array(img)
    
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    # image = image.transpose((2, 0, 1))
    
    # # numpy 转为MNN Tensor 数据类型时， 数据类型必须为float32, 否则会报错
    image = image.astype(np.float32)
    # construct tensor from np.ndarray
    tmp_input = MNN.Tensor((1, 3, INPUTSIZE, INPUTSIZE), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)

    return output_tensor.getData()
```


使用 衣服鞋子的多标签分类 数据集 进行衣服和鞋子的颜色和类型的分类， 
最后一层使用 sigmoid 激活函数。
并用 ncnn 进行推理测试

尝试使用 tensorrt-python 进行推理测试
`trt_inference.py`

### 结果
![](https://github.com/youngx123/Multi-label-classification/blob/main/eval_test/test_result/blue_jeans_31.png?raw=true)

![](https://github.com/youngx123/Multi-label-classification/blob/main/eval_test/test_result/blue_shirt_37.png?raw=true)

![](https://github.com/youngx123/Multi-label-classification/blob/main/eval_test/test_result/white_shoe_40.png?raw=true)
### 模型转化工具
>https://convertmodel.com/#outputFormat=tnn
