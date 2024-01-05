#pragma once
#include<iostream>
#include<string>

#include"cvLoad.h"
int main()
{
	const std::string netFile = R"(D:\MyNAS\Multi-label-classification-main\convert_model\multi_class_classification_sim.onnx)";
	std::string imgFile = R"(D:\MyNAS\Multi-label-classification-main\test_data\black_dress\15734780302_50.jpg)";
	const int imagesize = 416;
	cvLoad::onnxInference *model = new cvLoad::onnxInference(netFile, imagesize);


	model->Inference(imgFile);
	return 0;
}