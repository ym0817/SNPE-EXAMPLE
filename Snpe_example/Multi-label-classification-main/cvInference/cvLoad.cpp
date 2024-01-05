#include"cvLoad.h"
#include<opencv2/opencv.hpp>


cvLoad::onnxInference::onnxInference(const std::string &onxpath, const int size)
	:onnxPath(onxpath), imageSize(size)
{
	OnnxNet = cv::dnn::readNetFromONNX(onnxPath);
	if (OnnxNet.empty())
	{
		printf("load %s failed", onnxPath);
	}
}

void cvLoad::onnxInference::Inference(const std::string filedir)
{
	int index = filedir.rfind("\\");
	std::string baseName = filedir.substr(index + 1, filedir.size());
	std::cout << baseName << "\n";

	cv::Mat image = cv::imread(filedir);
	
	cv::Mat input, outpred;
	input = cv::dnn::blobFromImage(image, 1.0/255, cv::Size(imageSize, imageSize), cv::Scalar(0,0,0));
	
	OnnxNet.setInput(input);
	outpred = OnnxNet.forward("Predict");

	cv::Mat color_pred, type_pred;
	color_pred = outpred.colRange(0,4);
	type_pred = outpred.colRange(4, outpred.cols);

	int color_cls_index, type_cls_index;
	float color_max_prob, type_max_prob;


	argMax(color_pred, color_cls_index, color_max_prob);
	argMax(type_pred, type_cls_index, type_max_prob);

	std::string color_Name, type_Name;
	color_Name = COLOR.at(color_cls_index);
	type_Name = TYPE.at(type_cls_index);
	
	// draw and save
	std::string color_mess =  color_Name + ": "+ std::to_string(color_max_prob);
	cv::putText(image, color_mess, cv::Point(int(10), int(90)),
		cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

	std::string type_mess = type_Name + ": " + std::to_string(type_max_prob);
	cv::putText(image, type_mess, cv::Point(int(10), int(150)),
		cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

	cv::imwrite(baseName,image);

}


void cvLoad::onnxInference::argMax(cv::Mat &pred, int &index, float&score)
{
	score = -9999.0;
	float* data = pred.ptr<float>(0);
	for (int i = 0; i < pred.cols; i++)
	{
		if (score < data[i])
		{
			score = data[i];
			index = i;
		}
	}
}
