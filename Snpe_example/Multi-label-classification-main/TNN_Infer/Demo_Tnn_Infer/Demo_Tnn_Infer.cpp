// Demo_Tnn_Infer.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<opencv2/opencv.hpp>
#include<string>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/mat_utils.h"

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::string fdLoadFile(std::string path) 
{
	std::ifstream file(path, std::ios::in);
	//std::ifstream file(path, std::ios::binary);
	if (file.is_open()) {
		file.seekg(0, file.end);
		int size = file.tellg();
		char *content = new char[size];
		file.seekg(0, file.beg);
		file.read(content, size);
		std::string fileContent;
		fileContent.assign(content, size);
		delete[] content;
		file.close();
		return fileContent;
	}
	else {
		return "";
	}
}

int main()
{
	std::string model_buffer = "D:\\MyNAS\\Cpp_Inference\\Multi-label-classification-main\\TNN_Infer\\multi_label_classification_sim.opt.tnnmodel";
	std::string proto_buffer = "D:\\MyNAS\\Cpp_Inference\\Multi-label-classification-main\\TNN_Infer\\multi_label_classification_sim.opt.tnnproto";
	std::shared_ptr<tnn::TNN> net;
	std::shared_ptr<TNN_NS::Instance> instance;
	net = nullptr;
	if (net == nullptr)
	{
		std::string protoContent, modelContent;
		protoContent = fdLoadFile(proto_buffer);
		modelContent = fdLoadFile(model_buffer);

		TNN_NS::Status status;
		TNN_NS::ModelConfig config;
		config.model_type = TNN_NS::MODEL_TYPE_TNN;
		config.params = { protoContent, modelContent };
		auto net2 = std::make_shared<TNN_NS::TNN>();
		status = net2->Init(config);
		net = net2;

		TNN_NS::InputShapesMap shapeMap;
		TNN_NS::NetworkConfig network_config;
		network_config.library_path = { "" };
		network_config.device_type = TNN_NS::DEVICE_NAIVE;
		auto ins =  net->CreateInst(network_config, status, shapeMap);
		if (status != TNN_NS::TNN_OK || !ins) {
			network_config.device_type = TNN_NS::DEVICE_NAIVE;
			ins = net->CreateInst(network_config, status, shapeMap);
		}
		instance = ins;

		if (status != TNN_NS::TNN_OK) {
			std::cout << "load model error\n";
		}
		if (status == TNN_NS::TNN_OK) {
			std::cout<<"load model\n";
		}

	}

	std::string imgFile = "D:\\MyNAS\\Cpp_Inference\\test\\data\\test_data\\red_dress\\00000052.jpg";


	int image_width, image_height, image_channel;
	unsigned char *data = stbi_load(imgFile.c_str(), &image_height, &image_width, &image_channel, 3);
	std::cout << "image_height  " << image_height << " image_width  " << image_width << " image_channel  " << image_channel << std::endl;

	// 原始图片
	TNN_NS::DeviceType dt = TNN_NS::DEVICE_NAIVE;
	TNN_NS::DimsVector image_dims = { 1, 3, image_height, image_width };

	auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC3, image_dims, data);

	// 模型输入
	int size = 448;
	TNN_NS::DimsVector target_dims = { 1, 3, size, size };
	auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC3, target_dims);

	// OPENCL需要设置queue
	void *command_queue = nullptr;
	//auto status =  instance->GetCommandQueue(&command_queue);
	// 转换大小
	TNN_NS::ResizeParam param;
	param.type = TNN_NS::InterpType::INTERP_TYPE_NEAREST;
	auto status = TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, command_queue);
	// 输入数据
	TNN_NS::MatConvertParam input_cvt_param;
	input_cvt_param.scale = { 1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0 };
	input_cvt_param.bias = { 0.0, 0.0, 0.0, 0.0 };
	status =  instance->SetInputMat(resize_mat, input_cvt_param);
	status =  instance->Forward();
	// 获取数据
	std::vector<std::shared_ptr<TNN_NS::Mat>> output_mats;
	//for (const YoloLayerData &layerData : YoloV5::layers) {
	std::shared_ptr<TNN_NS::Mat> output_mat;
	TNN_NS::MatConvertParam param22;
	status =  instance->GetOutputMat(output_mat, param22, "outPutNode", TNN_NS::DEVICE_NAIVE);
	output_mats.push_back(output_mat);

	for (auto &output : output_mats) {
		auto dim = output->GetDims();
		auto *data = static_cast<float *>(output->GetData());
		int num_potential_detecs = dim[1] * dim[2];
		std::cout << num_potential_detecs<<std::endl;
		for (int i = 0; i < num_potential_detecs; ++i)
		{
    		// sigmoid在推理中完成
			float x = sigmoid(data[i]);
			std::cout << x << std::endl;
		}
	}
	system("pause");
	return 0;
}

