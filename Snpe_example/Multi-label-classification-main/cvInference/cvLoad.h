#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>


namespace cvLoad{
	class onnxInference
	{
	public:
		onnxInference(const std::string &onxpath, const int size);
		void Inference(const std::string filedir);
		void argMax(cv::Mat &pred, int &index, float&score);
	private:
		std::string onnxPath;
		int imageSize;
		cv::dnn::Net OnnxNet;

	public:
		std::vector<std::string>COLOR = { "white", "red", "blue", "black" };
		std::vector<std::string>TYPE = { "dress", "jeans", "shirt", "shoe", "bag" };
	};
}

